import os
import sys
import time
from datetime import datetime

import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
import matplotlib.pyplot as plt
import matplotlib
from IPython.display import clear_output
import numpy as np

# Optional GPU acceleration with CuPy
GPU_AVAILABLE = False
cp = None
try:
    import cupy as cp
    # Test if GPU is actually available
    cp.cuda.runtime.getDeviceCount()
    GPU_AVAILABLE = True
    print("GPU acceleration available via CuPy")
except ImportError:
    print("CuPy not installed - using CPU (install with: pip install cupy-cuda12x)")
except Exception as e:
    print(f"GPU not available: {e} - using CPU")

# Detect if running as standalone script
STANDALONE_MODE = __name__ == '__main__'

# Set matplotlib backend for interactive plotting
if STANDALONE_MODE:
    matplotlib.use('TkAgg')
    print("Running in STANDALONE mode - plots will open in separate windows")

class StreamingSimulator:
    """
    Simulates real-time data streaming from manufacturing robot controller.
    Reads CSV data point-by-point, stores in database, and visualizes in real-time.

    Data format: Current measurements from 14 robot axes with timestamps
    """

    def __init__(self, csv_path, db_config):
        """
        Initialize the streaming simulator.

        Parameters:
        -----------
        csv_path : str
            Path to the CSV file containing robot controller data
        db_config : dict
            Database configuration with keys: host, database, user, password
        """
        self.csv_path = csv_path
        self.db_config = db_config
        self.df = None
        self.current_index = 0
        self.data_buffer = []
        self.axis_columns = []

        # Figure references for reusing windows
        self._dashboard_fig = None
        self._dashboard_fig_num = None

        # Load CSV into memory
        self._load_csv()

        # Initialize database table
        self._init_database()
        
    def _load_csv(self):
        """Load CSV file into pandas DataFrame"""
        try:
            self.df = pd.read_csv(self.csv_path)
            
            # Identify axis columns
            self.axis_columns = [col for col in self.df.columns if col.startswith('Axis #')]
            
            # Convert Time to datetime
            self.df['Time'] = pd.to_datetime(self.df['Time'])
            
            print(f"✅ Loaded {len(self.df)} records from {self.csv_path}")
            print(f"📊 Columns: {list(self.df.columns)}")
            print(f"🔧 Detected {len(self.axis_columns)} robot axes")
            print(f"📅 Time range: {self.df['Time'].min()} to {self.df['Time'].max()}")
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            raise
    
    def _init_database(self):
        """Initialize database and create table if it doesn't exist"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Create table with columns for all 14 axes
            create_table_query = """
            CREATE TABLE IF NOT EXISTS robot_data (
                id SERIAL PRIMARY KEY,
                trait VARCHAR(50),
                axis_1 FLOAT,
                axis_2 FLOAT,
                axis_3 FLOAT,
                axis_4 FLOAT,
                axis_5 FLOAT,
                axis_6 FLOAT,
                axis_7 FLOAT,
                axis_8 FLOAT,
                axis_9 FLOAT,
                axis_10 FLOAT,
                axis_11 FLOAT,
                axis_12 FLOAT,
                axis_13 FLOAT,
                axis_14 FLOAT,
                timestamp TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            cursor.execute(create_table_query)
            
            # Create index on timestamp for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_robot_data_timestamp 
                ON robot_data(timestamp);
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            print("✅ Database table initialized")
        except Exception as e:
            print(f"❌ Database initialization error: {e}")
            raise
    
    def nextDataPoint(self, visualize=True, plot_style='combined'):
        """
        Read next data point from CSV, insert into database, and visualize.
        
        Parameters:
        -----------
        visualize : bool
            Whether to display visualization (default: True)
        plot_style : str
            'combined' - All axes on one graph (default)
            'separate' - Individual subplots for each axis
        
        Returns:
        --------
        dict : Current data point or None if end of data
        """
        if self.current_index >= len(self.df):
            print("⚠️ End of data stream reached")
            return None
        
        # Get current record
        record = self.df.iloc[self.current_index]
        
        # Insert into database
        self._insert_to_db(record)
        
        # Add to buffer for visualization
        self.data_buffer.append(record)
        
        # Visualize
        if visualize:
            if plot_style == 'combined':
                self._plot_dashboard(record)
            elif plot_style == 'separate':
                self._plot_dashboard_separate(record)
            else:
                print(f"⚠️ Unknown plot_style '{plot_style}'. Using 'combined'.")
                self._plot_dashboard(record)
        
        # Increment index
        self.current_index += 1

        return record.to_dict()

    def streamBatch(self, num_points, batch_size=500, visualize_every=1000, plot_style='combined'):
        """
        Stream multiple data points efficiently using batch database inserts.

        This is ~100x faster than nextDataPoint() for large datasets.

        Parameters:
        -----------
        num_points : int
            Number of data points to stream
        batch_size : int
            Number of records to insert per database batch (default: 500)
        visualize_every : int
            Visualize every Nth point (default: 1000, use 0 to disable)
        plot_style : str
            'combined' or 'separate' visualization style

        Returns:
        --------
        dict : Summary statistics of the streaming operation
        """
        start_time = time.time()
        points_to_process = min(num_points, len(self.df) - self.current_index)

        if points_to_process <= 0:
            print("⚠️ No more data to stream")
            return None

        print(f"🚀 Batch streaming {points_to_process:,} data points...")
        print(f"   Batch size: {batch_size}, Visualize every: {visualize_every}")

        # Timing accumulators for diagnostics
        time_db = 0
        time_viz = 0
        time_prep = 0

        # Open a single database connection for all inserts
        t0 = time.time()
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        print(f"   DB connection established in {time.time() - t0:.2f}s")

        points_processed = 0
        batches_inserted = 0
        current_batch = []

        # Pre-extract data using numpy arrays (FASTEST approach)
        t0 = time.time()
        end_idx = self.current_index + points_to_process
        slice_df = self.df.iloc[self.current_index:end_idx]

        # Extract columns as numpy arrays
        axis_cols = ['Axis #1', 'Axis #2', 'Axis #3', 'Axis #4', 'Axis #5', 'Axis #6', 'Axis #7',
                     'Axis #8', 'Axis #9', 'Axis #10', 'Axis #11', 'Axis #12', 'Axis #13', 'Axis #14']

        # Get trait values (or default) - convert to native Python strings
        if 'Trait' in slice_df.columns:
            traits = [str(v) if pd.notna(v) else 'current' for v in slice_df['Trait'].tolist()]
        else:
            traits = ['current'] * len(slice_df)

        # Get time values - convert to native Python datetime
        times = slice_df['Time'].tolist()

        # Get axis values as numpy arrays
        axis_arrays = []
        for col in axis_cols:
            if col in slice_df.columns:
                axis_arrays.append(slice_df[col].values)
            else:
                axis_arrays.append(np.full(len(slice_df), np.nan))

        # Build all values tuples with native Python types
        all_values = []
        for i in range(len(slice_df)):
            values = [traits[i]]
            for arr in axis_arrays:
                val = arr[i]
                values.append(float(val) if pd.notna(val) else None)
            # Convert timestamp to native Python datetime if needed
            time_val = times[i]
            if hasattr(time_val, 'to_pydatetime'):
                time_val = time_val.to_pydatetime()
            values.append(time_val)
            all_values.append(tuple(values))

        time_prep = time.time() - t0
        print(f"   Data prepared in {time_prep:.2f}s ({len(all_values):,} records)")

        try:
            # Insert in batches
            for batch_start in range(0, len(all_values), batch_size):
                batch_end = min(batch_start + batch_size, len(all_values))
                current_batch = all_values[batch_start:batch_end]

                t0 = time.time()
                self._insert_batch(cursor, current_batch)
                conn.commit()
                time_db += time.time() - t0

                batches_inserted += 1
                points_processed = batch_end

                # Progress update every 5 batches
                if batches_inserted % 5 == 0:
                    elapsed = time.time() - start_time
                    rate = points_processed / elapsed if elapsed > 0 else 0
                    remaining = (points_to_process - points_processed) / rate if rate > 0 else 0
                    print(f"   ✓ {points_processed:,}/{points_to_process:,} ({points_processed/points_to_process*100:.1f}%) "
                          f"- {rate:.0f} pts/sec - ETA: {remaining:.1f}s")

            # Add all records to buffer at once (for visualization/analysis)
            self.data_buffer.extend(slice_df.to_dict('records'))

            # Update current index
            self.current_index += points_processed

        finally:
            cursor.close()
            conn.close()

        elapsed_time = time.time() - start_time
        rate = points_processed / elapsed_time if elapsed_time > 0 else 0

        # Final visualization (only once at the end)
        if visualize_every > 0 and len(self.data_buffer) > 0:
            t0 = time.time()
            record = self.df.iloc[self.current_index - 1]
            if plot_style == 'combined':
                self._plot_dashboard_fast(record)
            else:
                self._plot_dashboard_separate(record)
            time_viz = time.time() - t0

        summary = {
            'points_processed': points_processed,
            'batches_inserted': batches_inserted,
            'elapsed_seconds': elapsed_time,
            'points_per_second': rate,
            'current_index': self.current_index,
            'buffer_size': len(self.data_buffer),
            'time_breakdown': {
                'data_prep': time_prep,
                'database': time_db,
                'visualization': time_viz
            }
        }

        print(f"\n✅ Batch streaming complete!")
        print(f"   Points processed: {points_processed:,}")
        print(f"   Time elapsed: {elapsed_time:.2f} seconds")
        print(f"   Rate: {rate:.0f} points/second")
        print(f"   Database batches: {batches_inserted}")
        print(f"\n   ⏱️ Time breakdown:")
        print(f"      Data prep:     {time_prep:.2f}s")
        print(f"      Database:      {time_db:.2f}s")
        print(f"      Visualization: {time_viz:.2f}s")

        return summary

    def _plot_dashboard_fast(self, current_record):
        """Lightweight visualization for batch streaming - uses sampling for large buffers"""
        if not STANDALONE_MODE:
            clear_output(wait=True)

        # Enable interactive mode for live updates
        plt.ion()

        # Reuse the same figure window to avoid multiple popups
        if self._dashboard_fig_num is not None and plt.fignum_exists(self._dashboard_fig_num):
            plt.figure(self._dashboard_fig_num)
            plt.clf()
            fig = plt.gcf()
        else:
            fig = plt.figure(figsize=(16, 8))
            self._dashboard_fig_num = fig.number
            self._dashboard_fig = fig

        # Sample buffer if too large (max 2000 points for visualization)
        buffer_size = len(self.data_buffer)
        if buffer_size > 2000:
            # Use last 2000 points only for visualization
            sample_indices = list(range(max(0, buffer_size - 2000), buffer_size))
            buffer_sample = [self.data_buffer[i] for i in sample_indices]
        else:
            buffer_sample = self.data_buffer

        buffer_df = pd.DataFrame(buffer_sample)

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A',
                 '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2']

        ax = fig.add_subplot(111)

        # Plot only first 8 axes for speed
        for idx, axis in enumerate(self.axis_columns[:8]):
            if axis in buffer_df.columns:
                ax.plot(buffer_df[axis].values, color=colors[idx], linewidth=1, alpha=0.7, label=axis)

        ax.set_title(f'Robot Current Monitoring - {self.current_index:,} points streamed', fontsize=14, fontweight='bold')
        ax.set_xlabel('Sample Index (last 2000 points)', fontsize=11)
        ax.set_ylabel('Current (A)', fontsize=11)
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        # Add progress text
        progress_pct = (self.current_index / len(self.df)) * 100
        ax.text(0.02, 0.98, f'Progress: {self.current_index:,}/{len(self.df):,} ({progress_pct:.1f}%)',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plt.tight_layout()
        plt.show()

    def _insert_batch(self, cursor, batch):
        """Insert a batch of records using execute_values for maximum speed"""
        insert_query = """
        INSERT INTO robot_data
        (trait, axis_1, axis_2, axis_3, axis_4, axis_5, axis_6, axis_7,
         axis_8, axis_9, axis_10, axis_11, axis_12, axis_13, axis_14, timestamp)
        VALUES %s
        """
        execute_values(cursor, insert_query, batch, page_size=len(batch))

    def _insert_to_db(self, record):
        """Insert a single record into the database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Prepare INSERT query
            insert_query = """
            INSERT INTO robot_data 
            (trait, axis_1, axis_2, axis_3, axis_4, axis_5, axis_6, axis_7, 
             axis_8, axis_9, axis_10, axis_11, axis_12, axis_13, axis_14, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            # Extract values - handle NaN values
            values = (
                record.get('Trait', 'current'),
                float(record['Axis #1']) if pd.notna(record.get('Axis #1')) else None,
                float(record['Axis #2']) if pd.notna(record.get('Axis #2')) else None,
                float(record['Axis #3']) if pd.notna(record.get('Axis #3')) else None,
                float(record['Axis #4']) if pd.notna(record.get('Axis #4')) else None,
                float(record['Axis #5']) if pd.notna(record.get('Axis #5')) else None,
                float(record['Axis #6']) if pd.notna(record.get('Axis #6')) else None,
                float(record['Axis #7']) if pd.notna(record.get('Axis #7')) else None,
                float(record['Axis #8']) if pd.notna(record.get('Axis #8')) else None,
                float(record['Axis #9']) if pd.notna(record.get('Axis #9')) else None,
                float(record['Axis #10']) if pd.notna(record.get('Axis #10')) else None,
                float(record['Axis #11']) if pd.notna(record.get('Axis #11')) else None,
                float(record['Axis #12']) if pd.notna(record.get('Axis #12')) else None,
                float(record['Axis #13']) if pd.notna(record.get('Axis #13')) else None,
                float(record['Axis #14']) if pd.notna(record.get('Axis #14')) else None,
                record['Time']
            )
            
            cursor.execute(insert_query, values)
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"❌ Database insert error at index {self.current_index}: {e}")
    
    def _plot_dashboard(self, current_record):
        """Real-time visualization dashboard - all axes on one graph"""
        # Only clear output in notebook mode
        if not STANDALONE_MODE:
            clear_output(wait=True)

        # Enable interactive mode for live updates
        plt.ion()

        # Reuse the same figure window to avoid multiple popups
        if self._dashboard_fig_num is not None and plt.fignum_exists(self._dashboard_fig_num):
            plt.figure(self._dashboard_fig_num)
            plt.clf()
            fig = plt.gcf()
        else:
            fig = plt.figure(figsize=(18, 10))
            self._dashboard_fig_num = fig.number
            self._dashboard_fig = fig
        
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.25)
        
        fig.suptitle('Robot Controller Current Monitoring - Real-Time Stream (All Axes)', 
                     fontsize=16, fontweight='bold')
        
        if len(self.data_buffer) > 0:
            buffer_df = pd.DataFrame(self.data_buffer)
            
            # Main plot - All axes on one graph
            ax_main = fig.add_subplot(gs[0])
            
            # Define colors for all 14 axes
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', 
                     '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2',
                     '#F06292', '#AED581', '#FFD54F', '#90CAF9',
                     '#FFAB91', '#CE93D8']
            
            # Use Time for x-axis
            time_data = buffer_df['Time']
            
            # Plot all axes
            lines = []
            labels = []
            for idx, axis in enumerate(self.axis_columns):
                if axis in buffer_df.columns:
                    axis_data = buffer_df[axis].dropna()
                    
                    if len(axis_data) > 0:
                        # Plot line using Time as x-axis
                        line, = ax_main.plot(time_data, buffer_df[axis], 
                                           color=colors[idx % len(colors)], 
                                           linewidth=2, alpha=0.7, 
                                           label=axis)
                        lines.append(line)
                        labels.append(axis)
                        
                        # Highlight current point
                        current_val = current_record.get(axis, 0)
                        if pd.notna(current_val) and current_val > 0:
                            ax_main.scatter(current_record['Time'], current_val, 
                                          color=colors[idx % len(colors)], 
                                          s=80, zorder=5, 
                                          edgecolors='black', linewidth=1.5)
            
            # Formatting
            ax_main.set_title('Current Draw - All Robot Axes', 
                            fontsize=14, fontweight='bold', pad=15)
            ax_main.set_xlabel('Time', fontsize=12)
            ax_main.set_ylabel('Current (Amperes)', fontsize=12)
            ax_main.grid(True, alpha=0.3, linestyle=':', linewidth=1)
            
            # Format x-axis to show time nicely
            import matplotlib.dates as mdates
            ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax_main.tick_params(axis='x', rotation=45)
            
            # Legend - split into two columns if many axes
            if len(labels) > 7:
                ax_main.legend(lines, labels, loc='upper left', 
                             fontsize=9, ncol=2, framealpha=0.9,
                             bbox_to_anchor=(0, 1), borderaxespad=0)
            else:
                ax_main.legend(lines, labels, loc='upper left', 
                             fontsize=9, framealpha=0.9)
            
            # Add reference line for total current
            total_current_series = buffer_df[self.axis_columns].sum(axis=1, skipna=True)
            ax_main_twin = ax_main.twinx()
            ax_main_twin.plot(time_data, total_current_series, 
                            color='black', linewidth=2.5, alpha=0.4, 
                            linestyle='--', label='Total Current')
            ax_main_twin.set_ylabel('Total Current (A)', fontsize=11, color='black')
            ax_main_twin.tick_params(axis='y', labelcolor='black')
            ax_main_twin.legend(loc='upper right', fontsize=9, framealpha=0.9)
            
            # Summary statistics panel
            ax_summary = fig.add_subplot(gs[1])
            ax_summary.axis('off')
            
            # Calculate detailed stats
            active_axes_count = sum(1 for ax in self.axis_columns 
                                  if ax in buffer_df.columns 
                                  and pd.notna(buffer_df[ax].iloc[-1]) 
                                  and buffer_df[ax].iloc[-1] > 0.1)
            
            total_current = total_current_series.iloc[-1]
            avg_total_current = total_current_series.mean()
            max_single_axis = max((buffer_df[ax].iloc[-1] for ax in self.axis_columns 
                                 if ax in buffer_df.columns and pd.notna(buffer_df[ax].iloc[-1])), 
                                default=0)
            
            # Find axis with max current
            max_axis = None
            max_axis_val = 0
            for ax in self.axis_columns:
                if ax in buffer_df.columns and pd.notna(buffer_df[ax].iloc[-1]):
                    val = buffer_df[ax].iloc[-1]
                    if val > max_axis_val:
                        max_axis_val = val
                        max_axis = ax
            
            summary_text = f"""
    📊 REAL-TIME STREAMING STATUS
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Progress: {self.current_index + 1:,} / {len(self.df):,} records ({((self.current_index + 1) / len(self.df) * 100):.1f}%)  |  Timestamp: {current_record['Time']}
    
    Current Metrics:                              Statistical Summary:
      • Total Current:     {total_current:>6.2f} A          • Avg Total Current:  {avg_total_current:>6.2f} A
      • Max Axis Current:  {max_axis_val:>6.2f} A ({max_axis})    • Active Axes:       {active_axes_count:>2d} / {len(self.axis_columns)}
      • Buffer Size:       {len(buffer_df):>6,} pts         • Data Quality:       {(1 - buffer_df[self.axis_columns].isna().sum().sum() / (len(buffer_df) * len(self.axis_columns))) * 100:.1f}%
            """
            
            ax_summary.text(0.02, 0.5, summary_text,
                          fontsize=9.5, family='monospace',
                          verticalalignment='center',
                          bbox=dict(boxstyle='round', facecolor='lightblue',
                                  alpha=0.3, edgecolor='steelblue', linewidth=2))

        # Use tight_layout with warning suppression (axis('off') and twinx cause warnings)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plt.tight_layout()

        # Different display method for standalone vs notebook
        if STANDALONE_MODE:
            plt.draw()
            plt.pause(0.001)
            # Print progress to console
            print(f"\r📊 Streaming: {self.current_index + 1}/{len(self.df)} ({((self.current_index + 1) / len(self.df) * 100):.1f}%)", end='', flush=True)
        else:
            plt.show()
    
    def _plot_dashboard_separate(self, current_record):
        """Real-time visualization dashboard - separate subplots for each axis"""
        if not STANDALONE_MODE:
            clear_output(wait=True)

        # Enable interactive mode for live updates
        plt.ion()

        # Reuse the same figure window to avoid multiple popups
        if self._dashboard_fig_num is not None and plt.fignum_exists(self._dashboard_fig_num):
            plt.figure(self._dashboard_fig_num)
            plt.clf()
            fig = plt.gcf()
        else:
            fig = plt.figure(figsize=(18, 12))
            self._dashboard_fig_num = fig.number
            self._dashboard_fig = fig
        gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
        
        fig.suptitle('Robot Controller Current Monitoring - Individual Axes View', 
                     fontsize=16, fontweight='bold')
        
        if len(self.data_buffer) > 0:
            buffer_df = pd.DataFrame(self.data_buffer)
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', 
                     '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2',
                     '#F06292', '#AED581', '#FFD54F', '#90CAF9',
                     '#FFAB91', '#CE93D8']
            
            # Use Time for x-axis
            time_data = buffer_df['Time']
            
            # Import for time formatting
            import matplotlib.dates as mdates
            
            # Plot each axis in separate subplot
            for idx, axis in enumerate(self.axis_columns):
                if idx >= 14:  # Only plot first 14 axes
                    break
                    
                row = idx // 4
                col = idx % 4
                ax = fig.add_subplot(gs[row, col])
                
                if axis in buffer_df.columns:
                    axis_data = buffer_df[axis].dropna()
                    
                    if len(axis_data) > 0:
                        # Plot line using Time as x-axis
                        ax.plot(time_data, buffer_df[axis], 
                               color=colors[idx], linewidth=2, alpha=0.7)
                        
                        # Highlight current point
                        current_val = current_record.get(axis, 0)
                        if pd.notna(current_val) and current_val > 0:
                            ax.scatter(current_record['Time'], current_val, 
                                      color='red', s=80, zorder=5, 
                                      edgecolors='black', linewidth=1.5)
                        
                        # Add mean line
                        mean_val = axis_data.mean()
                        if pd.notna(mean_val):
                            ax.axhline(mean_val, color='green', linestyle='--', 
                                      linewidth=1, alpha=0.5, 
                                      label=f'μ={mean_val:.2f}A')
                        
                        ax.set_title(f'{axis}', fontsize=10, fontweight='bold')
                        ax.set_xlabel('Time', fontsize=8)
                        ax.set_ylabel('Current (A)', fontsize=8)
                        ax.grid(True, alpha=0.3, linestyle=':')
                        ax.legend(fontsize=7, loc='upper right')
                        ax.tick_params(labelsize=7, axis='x', rotation=45)

                        # Format x-axis
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plt.tight_layout()
        plt.show()
    
    def get_all_data_from_db(self):
        """Retrieve all data from database for analysis"""
        try:
            conn = psycopg2.connect(**self.db_config)
            query = "SELECT * FROM robot_data ORDER BY timestamp"
            df = pd.read_sql(query, conn)
            conn.close()
            print(f"Retrieved {len(df)} records from database")
            return df
        except Exception as e:
            print(f"Error retrieving data: {e}")
            return None

    def fetch_next_from_db(self, visualize=True, plot_style='combined'):
        """
        Fetch the next record from the database one at a time and visualize.

        This method queries the database for a single record based on the current index,
        adds it to the visualization buffer, and displays the updated plot.

        Parameters:
        -----------
        visualize : bool
            Whether to display visualization (default: True)
        plot_style : str
            'combined' - All axes on one graph (default)
            'separate' - Individual subplots for each axis

        Returns:
        --------
        dict : Current data point or None if end of data
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            # Fetch one record at the current offset
            query = """
            SELECT id, trait, axis_1, axis_2, axis_3, axis_4, axis_5, axis_6, axis_7,
                   axis_8, axis_9, axis_10, axis_11, axis_12, axis_13, axis_14, timestamp
            FROM robot_data
            ORDER BY id
            LIMIT 1 OFFSET %s
            """
            cursor.execute(query, (self.current_index,))
            row = cursor.fetchone()

            cursor.close()
            conn.close()

            if row is None:
                print("End of database records reached")
                return None

            # Convert to dictionary matching original format
            record = {
                'Trait': row[1],
                'Axis #1': row[2],
                'Axis #2': row[3],
                'Axis #3': row[4],
                'Axis #4': row[5],
                'Axis #5': row[6],
                'Axis #6': row[7],
                'Axis #7': row[8],
                'Axis #8': row[9],
                'Axis #9': row[10],
                'Axis #10': row[11],
                'Axis #11': row[12],
                'Axis #12': row[13],
                'Axis #13': row[14],
                'Axis #14': row[15],
                'Time': row[16]
            }

            # Add to buffer for visualization
            self.data_buffer.append(record)

            # Convert to pandas Series for plotting methods
            record_series = pd.Series(record)

            # Visualize
            if visualize:
                if plot_style == 'combined':
                    self._plot_dashboard(record_series)
                elif plot_style == 'separate':
                    self._plot_dashboard_separate(record_series)
                else:
                    self._plot_dashboard(record_series)

            # Increment index
            self.current_index += 1

            return record

        except Exception as e:
            print(f"Error fetching from database at index {self.current_index}: {e}")
            return None

    def stream_from_db(self, visualize_every=1, plot_style='combined', fetch_size=1):
        """
        Stream the entire dataset from the database with real-time visualization.

        This method is synced with the notebook's single-point streaming logic.
        Uses a single database connection with server-side cursor while keeping
        the GUI responsive. Close the plot window to stop streaming.

        Single-Point Streaming Mode (fetch_size=1, visualize_every=1):
        - Fetches exactly 1 record per database call
        - Updates visualization after every single point
        - Provides true real-time streaming experience
        - Rate: ~10-50 points/second depending on system

        Batch Streaming Mode (fetch_size>1):
        - Fetches multiple records per database call
        - Faster throughput but less "real-time" feel
        - Use for processing large datasets quickly

        Parameters:
        -----------
        visualize_every : int
            Visualize every Nth point (default: 1 for every point)
        plot_style : str
            'combined' - All axes on one graph (default)
            'separate' - Individual subplots for each axis
        fetch_size : int
            Number of records to fetch per database call (default: 1 for true real-time)
            Use higher values (e.g., 100) for faster bulk streaming

        Returns:
        --------
        dict : Summary statistics including points_processed, elapsed_seconds,
               points_per_second, buffer_size, and stopped_early flag
        """
        start_time = time.time()

        # Use a single connection for the entire streaming operation
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM robot_data")
            total_records = cursor.fetchone()[0]
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return None

        if total_records == 0:
            print("No records in database to stream")
            conn.close()
            return None

        if fetch_size == 1 and visualize_every == 1:
            print(f"Starting SINGLE-POINT database stream: {total_records:,} records")
            print("Mode: True real-time (1 record fetched and visualized at a time)")
        else:
            print(f"Starting database stream: {total_records:,} records")
            print(f"Fetch size: {fetch_size} record(s), Visualize every: {visualize_every} point(s)")
        print("Plots will open in a separate window...")
        print("Close the plot window or press Ctrl+C to stop\n")

        # Enable interactive mode for live updates
        plt.ion()

        points_processed = 0
        batch_size = fetch_size  # Use user-specified fetch size
        user_stopped = False  # Flag to track if user closed the window

        try:
            # Use server-side cursor for memory-efficient streaming
            cursor = conn.cursor(name='streaming_cursor')
            cursor.itersize = batch_size

            query = """
            SELECT id, trait, axis_1, axis_2, axis_3, axis_4, axis_5, axis_6, axis_7,
                   axis_8, axis_9, axis_10, axis_11, axis_12, axis_13, axis_14, timestamp
            FROM robot_data
            ORDER BY id
            """
            cursor.execute(query)

            # Pre-create the figure for reuse
            if self._dashboard_fig_num is None or not plt.fignum_exists(self._dashboard_fig_num):
                fig = plt.figure(figsize=(18, 10))
                self._dashboard_fig_num = fig.number
                self._dashboard_fig = fig

            last_viz_time = time.time()
            min_viz_interval = 0.05  # Minimum 50ms between visualizations

            while True:
                # Check if user closed the plot window
                if self._dashboard_fig_num is not None and not plt.fignum_exists(self._dashboard_fig_num):
                    print("\n\nPlot window closed - stopping stream...")
                    user_stopped = True
                    break

                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break

                for row in rows:
                    # Check again inside the inner loop for responsiveness
                    if self._dashboard_fig_num is not None and not plt.fignum_exists(self._dashboard_fig_num):
                        user_stopped = True
                        break
                    # Convert to dictionary matching original format
                    record = {
                        'Trait': row[1],
                        'Axis #1': row[2],
                        'Axis #2': row[3],
                        'Axis #3': row[4],
                        'Axis #4': row[5],
                        'Axis #5': row[6],
                        'Axis #6': row[7],
                        'Axis #7': row[8],
                        'Axis #8': row[9],
                        'Axis #9': row[10],
                        'Axis #10': row[11],
                        'Axis #11': row[12],
                        'Axis #12': row[13],
                        'Axis #13': row[14],
                        'Axis #14': row[15],
                        'Time': row[16]
                    }

                    # Add to buffer
                    self.data_buffer.append(record)
                    points_processed += 1
                    self.current_index += 1

                    # Visualize at specified intervals
                    should_visualize = (points_processed % visualize_every == 0)
                    current_time = time.time()

                    # For single-point streaming (fetch_size=1), always visualize and process events
                    if batch_size == 1:
                        if should_visualize:
                            record_series = pd.Series(record)
                            if plot_style == 'combined':
                                self._plot_dashboard_nonblocking(record_series)
                            else:
                                self._plot_dashboard_separate_nonblocking(record_series)
                        # Always process GUI events for responsive close button
                        plt.pause(0.001)
                    else:
                        # Batch mode: throttle visualization to prevent GUI freeze
                        if should_visualize and (current_time - last_viz_time) >= min_viz_interval:
                            record_series = pd.Series(record)
                            if plot_style == 'combined':
                                self._plot_dashboard_nonblocking(record_series)
                            else:
                                self._plot_dashboard_separate_nonblocking(record_series)
                            last_viz_time = current_time

                        # Process GUI events to keep window responsive (every 10 points)
                        if points_processed % 10 == 0:
                            plt.pause(0.001)

                    # Progress update frequency based on fetch size
                    progress_interval = 100 if batch_size == 1 else 500
                    if points_processed % progress_interval == 0:
                        elapsed = time.time() - start_time
                        rate = points_processed / elapsed if elapsed > 0 else 0
                        remaining = (total_records - points_processed) / rate if rate > 0 else 0
                        print(f"\rStreamed: {points_processed:,}/{total_records:,} ({points_processed/total_records*100:.1f}%) - {rate:.1f} pts/sec - ETA: {remaining:.1f}s", end='', flush=True)

                # Break outer loop if user closed window
                if user_stopped:
                    break

            cursor.close()
            conn.close()

        except KeyboardInterrupt:
            print("\n\nStreaming interrupted by user (Ctrl+C)")
            user_stopped = True
            try:
                cursor.close()
                conn.close()
            except:
                pass

        except Exception as e:
            print(f"\nError during streaming: {e}")
            try:
                cursor.close()
                conn.close()
            except:
                pass

        elapsed_time = time.time() - start_time
        rate = points_processed / elapsed_time if elapsed_time > 0 else 0

        summary = {
            'points_processed': points_processed,
            'total_records': total_records,
            'elapsed_seconds': elapsed_time,
            'points_per_second': rate,
            'buffer_size': len(self.data_buffer),
            'stopped_early': user_stopped
        }

        if user_stopped:
            print(f"\n\nStreaming stopped by user")
        else:
            print(f"\n\nStreaming complete!")
        print(f"   Points processed: {points_processed:,}/{total_records:,}")
        print(f"   Time elapsed: {elapsed_time:.2f} seconds")
        print(f"   Rate: {rate:.0f} points/second")

        return summary

    def _plot_dashboard_nonblocking(self, current_record):
        """
        Non-blocking visualization that keeps the GUI responsive.
        Synced with notebook's single-point streaming visualization logic.
        Uses Time on x-axis for consistency with _plot_dashboard.
        """
        import matplotlib.dates as mdates

        # Reuse existing figure
        if self._dashboard_fig_num is not None and plt.fignum_exists(self._dashboard_fig_num):
            fig = plt.figure(self._dashboard_fig_num)
            fig.clf()
        else:
            fig = plt.figure(figsize=(18, 10))
            self._dashboard_fig_num = fig.number
            self._dashboard_fig = fig

        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.25)
        fig.suptitle('Robot Controller Current Monitoring - Real-Time Stream (Single Point)',
                     fontsize=16, fontweight='bold')

        if len(self.data_buffer) > 0:
            # Limit buffer size for visualization to prevent slowdown
            max_viz_points = 2000
            if len(self.data_buffer) > max_viz_points:
                viz_buffer = self.data_buffer[-max_viz_points:]
            else:
                viz_buffer = self.data_buffer

            buffer_df = pd.DataFrame(viz_buffer)

            # Main plot
            ax_main = fig.add_subplot(gs[0])

            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A',
                     '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2']

            # Use Time for x-axis (consistent with notebook)
            time_data = buffer_df['Time'] if 'Time' in buffer_df.columns else buffer_df.index

            # Plot first 8 axes (primary axes as per notebook)
            lines = []
            labels = []
            for idx, axis in enumerate(self.axis_columns[:8]):
                if axis in buffer_df.columns:
                    axis_data = buffer_df[axis]
                    line, = ax_main.plot(time_data, axis_data,
                               color=colors[idx % len(colors)],
                               linewidth=1.5, alpha=0.7, label=axis)
                    lines.append(line)
                    labels.append(axis)

                    # Highlight current point (last point in buffer)
                    current_val = axis_data.iloc[-1] if len(axis_data) > 0 else None
                    if current_val is not None and pd.notna(current_val) and current_val > 0:
                        current_time = time_data.iloc[-1] if hasattr(time_data, 'iloc') else time_data[-1]
                        ax_main.scatter(current_time, current_val,
                                      color=colors[idx % len(colors)],
                                      s=80, zorder=5,
                                      edgecolors='black', linewidth=1.5)

            ax_main.set_title(f'Current Draw - Primary 8 Axes (Last {len(viz_buffer):,} points)',
                            fontsize=14, fontweight='bold', pad=10)
            ax_main.set_xlabel('Time', fontsize=12)
            ax_main.set_ylabel('Current (Amperes)', fontsize=12)
            ax_main.grid(True, alpha=0.3, linestyle=':', linewidth=1)
            ax_main.legend(loc='upper left', fontsize=9, ncol=2, framealpha=0.9)

            # Format x-axis to show time nicely
            if 'Time' in buffer_df.columns:
                ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                ax_main.tick_params(axis='x', rotation=45)

            # Add total current on secondary y-axis (consistent with _plot_dashboard)
            total_current_series = buffer_df[self.axis_columns[:8]].sum(axis=1, skipna=True)
            ax_twin = ax_main.twinx()
            ax_twin.plot(time_data, total_current_series,
                        color='black', linewidth=2.5, alpha=0.4,
                        linestyle='--', label='Total Current')
            ax_twin.set_ylabel('Total Current (A)', fontsize=11, color='black')
            ax_twin.tick_params(axis='y', labelcolor='black')
            ax_twin.legend(loc='upper right', fontsize=9, framealpha=0.9)

            # Summary panel
            ax_summary = fig.add_subplot(gs[1])
            ax_summary.axis('off')

            total_current = total_current_series.iloc[-1] if len(total_current_series) > 0 else 0
            avg_total_current = total_current_series.mean() if len(total_current_series) > 0 else 0

            # Count active axes
            active_axes_count = sum(1 for ax in self.axis_columns[:8]
                                  if ax in buffer_df.columns
                                  and len(buffer_df[ax].dropna()) > 0
                                  and buffer_df[ax].iloc[-1] > 0.1)

            # Find axis with max current
            max_axis = None
            max_axis_val = 0
            for ax in self.axis_columns[:8]:
                if ax in buffer_df.columns and pd.notna(buffer_df[ax].iloc[-1]):
                    val = buffer_df[ax].iloc[-1]
                    if val > max_axis_val:
                        max_axis_val = val
                        max_axis = ax

            # Get current timestamp
            current_timestamp = current_record.get('Time', 'N/A') if hasattr(current_record, 'get') else current_record['Time']

            summary_text = f"""
    📊 REAL-TIME STREAMING STATUS (Single Point Mode)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Progress: {self.current_index:,} / {len(self.df):,} records ({(self.current_index / len(self.df) * 100):.1f}%)  |  Timestamp: {current_timestamp}

    Current Metrics:                              Statistical Summary:
      • Total Current:     {total_current:>6.2f} A          • Avg Total Current:  {avg_total_current:>6.2f} A
      • Max Axis Current:  {max_axis_val:>6.2f} A ({max_axis or 'N/A'})    • Active Axes:       {active_axes_count:>2d} / 8
      • Buffer Size:       {len(self.data_buffer):>6,} pts         • Streaming Mode:     1 point at a time
            """

            ax_summary.text(0.02, 0.5, summary_text,
                          fontsize=9.5, family='monospace',
                          verticalalignment='center',
                          bbox=dict(boxstyle='round', facecolor='lightblue',
                                  alpha=0.3, edgecolor='steelblue', linewidth=2))

        # Use tight_layout with warning suppression (axis('off') and twinx cause warnings)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig.tight_layout()
        fig.canvas.draw_idle()  # Non-blocking draw
        fig.canvas.flush_events()  # Process pending events

    def _plot_dashboard_separate_nonblocking(self, current_record):
        """
        Non-blocking separate axes visualization.
        Synced with notebook's visualization logic - uses Time on x-axis.
        """
        import matplotlib.dates as mdates

        if self._dashboard_fig_num is not None and plt.fignum_exists(self._dashboard_fig_num):
            fig = plt.figure(self._dashboard_fig_num)
            fig.clf()
        else:
            fig = plt.figure(figsize=(18, 12))
            self._dashboard_fig_num = fig.number
            self._dashboard_fig = fig

        gs = fig.add_gridspec(2, 4, hspace=0.4, wspace=0.3)
        fig.suptitle(f'Robot Controller - Individual Axes View ({self.current_index:,} points streamed)',
                     fontsize=16, fontweight='bold')

        if len(self.data_buffer) > 0:
            max_viz_points = 2000
            if len(self.data_buffer) > max_viz_points:
                viz_buffer = self.data_buffer[-max_viz_points:]
            else:
                viz_buffer = self.data_buffer

            buffer_df = pd.DataFrame(viz_buffer)

            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A',
                     '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2']

            # Use Time for x-axis (consistent with notebook)
            time_data = buffer_df['Time'] if 'Time' in buffer_df.columns else buffer_df.index

            for idx, axis in enumerate(self.axis_columns[:8]):
                row = idx // 4
                col = idx % 4
                ax = fig.add_subplot(gs[row, col])

                if axis in buffer_df.columns:
                    axis_data = buffer_df[axis]
                    ax.plot(time_data, axis_data, color=colors[idx], linewidth=1.5, alpha=0.7)

                    # Highlight current point
                    current_val = axis_data.iloc[-1] if len(axis_data) > 0 else None
                    if current_val is not None and pd.notna(current_val) and current_val > 0:
                        current_time = time_data.iloc[-1] if hasattr(time_data, 'iloc') else time_data[-1]
                        ax.scatter(current_time, current_val,
                                  color='red', s=60, zorder=5,
                                  edgecolors='black', linewidth=1.5)

                    # Add mean line
                    axis_clean = axis_data.dropna()
                    if len(axis_clean) > 0:
                        mean_val = axis_clean.mean()
                        ax.axhline(mean_val, color='green', linestyle='--',
                                  linewidth=1, alpha=0.5,
                                  label=f'μ={mean_val:.2f}A')
                        ax.legend(fontsize=7, loc='upper right')

                    ax.set_title(f'{axis}', fontsize=10, fontweight='bold')
                    ax.set_xlabel('Time', fontsize=8)
                    ax.set_ylabel('Current (A)', fontsize=8)
                    ax.grid(True, alpha=0.3, linestyle=':')
                    ax.tick_params(labelsize=7)

                    # Format x-axis
                    if 'Time' in buffer_df.columns:
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                        ax.tick_params(axis='x', rotation=45)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig.tight_layout()
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    def get_db_record_count(self):
        """Get the total number of records in the database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM robot_data")
            count = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            return count
        except Exception as e:
            print(f"Error getting record count: {e}")
            return 0
    
    def detect_anomalies(self, threshold_multiplier=2.5, use_gpu=False):
        """
        Detect anomalies in robot axis currents using statistical methods.
        Optionally uses GPU acceleration for faster computation on large datasets.

        Parameters:
        -----------
        threshold_multiplier : float
            Number of standard deviations for anomaly detection
        use_gpu : bool
            Whether to use GPU acceleration if available (default: False)

        Returns:
        --------
        dict : Dictionary of anomalies by axis
        """
        if len(self.data_buffer) < 30:
            print("Not enough data for reliable anomaly detection (need 30+ points)")
            return {}

        buffer_df = pd.DataFrame(self.data_buffer)
        anomalies = {}

        # Use GPU if requested and available
        if use_gpu and GPU_AVAILABLE:
            return self._detect_anomalies_gpu(buffer_df, threshold_multiplier)

        for axis in self.axis_columns:
            if axis in buffer_df.columns:
                # Get non-null values
                axis_data = buffer_df[axis].dropna()

                if len(axis_data) < 10:
                    continue

                mean = axis_data.mean()
                std = axis_data.std()

                if std == 0:  # Avoid division by zero
                    continue

                threshold_upper = mean + (threshold_multiplier * std)
                threshold_lower = mean - (threshold_multiplier * std)

                # Find anomalies
                anomaly_mask = (buffer_df[axis] > threshold_upper) | (buffer_df[axis] < threshold_lower)
                anomaly_indices = buffer_df[anomaly_mask].index.tolist()

                if anomaly_indices:
                    anomalies[axis] = {
                        'indices': anomaly_indices,
                        'values': buffer_df.loc[anomaly_indices, axis].tolist(),
                        'mean': mean,
                        'std': std,
                        'threshold_upper': threshold_upper,
                        'threshold_lower': threshold_lower,
                        'severity': 'HIGH' if len(anomaly_indices) > 5 else 'MEDIUM'
                    }

        return anomalies

    def _detect_anomalies_gpu(self, buffer_df, threshold_multiplier):
        """GPU-accelerated anomaly detection using CuPy"""
        anomalies = {}

        for axis in self.axis_columns:
            if axis not in buffer_df.columns:
                continue

            # Get data and move to GPU
            axis_data = buffer_df[axis].values
            valid_mask = ~np.isnan(axis_data)

            if valid_mask.sum() < 10:
                continue

            # Transfer to GPU
            gpu_data = cp.asarray(axis_data)
            gpu_valid_mask = cp.asarray(valid_mask)

            # Compute statistics on GPU
            valid_data = gpu_data[gpu_valid_mask]
            mean = float(cp.mean(valid_data))
            std = float(cp.std(valid_data))

            if std == 0:
                continue

            threshold_upper = mean + (threshold_multiplier * std)
            threshold_lower = mean - (threshold_multiplier * std)

            # Find anomalies on GPU
            anomaly_mask = (gpu_data > threshold_upper) | (gpu_data < threshold_lower)
            anomaly_mask = anomaly_mask & gpu_valid_mask

            # Transfer results back to CPU
            anomaly_mask_cpu = cp.asnumpy(anomaly_mask)
            anomaly_indices = np.where(anomaly_mask_cpu)[0].tolist()

            if anomaly_indices:
                anomalies[axis] = {
                    'indices': anomaly_indices,
                    'values': axis_data[anomaly_mask_cpu].tolist(),
                    'mean': mean,
                    'std': std,
                    'threshold_upper': threshold_upper,
                    'threshold_lower': threshold_lower,
                    'severity': 'HIGH' if len(anomaly_indices) > 5 else 'MEDIUM'
                }

        return anomalies

    def stream_from_db_gpu(self, visualize_every=10, plot_style='combined', batch_size=1000):
        """
        GPU-accelerated database streaming with batch processing.

        Fetches data in batches and processes on GPU for maximum throughput.
        This is the fastest option for large datasets when GPU is available.

        Parameters:
        -----------
        visualize_every : int
            Visualize every Nth batch (default: 10)
        plot_style : str
            'combined' or 'separate' visualization style
        batch_size : int
            Number of records to fetch per batch (default: 1000)

        Returns:
        --------
        dict : Summary statistics
        """
        if not GPU_AVAILABLE:
            print("GPU not available, falling back to CPU streaming...")
            return self.stream_from_db(visualize_every=visualize_every, plot_style=plot_style)

        start_time = time.time()

        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM robot_data")
            total_records = cursor.fetchone()[0]
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return None

        if total_records == 0:
            print("No records in database to stream")
            conn.close()
            return None

        print(f"Starting GPU-accelerated database stream: {total_records:,} records")
        print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
        print("Close the plot window or press Ctrl+C to stop\n")

        plt.ion()
        points_processed = 0
        batches_processed = 0
        user_stopped = False  # Flag to track if user closed the window

        # Pre-allocate GPU arrays for batch processing
        gpu_batch_data = None

        try:
            cursor = conn.cursor(name='gpu_streaming_cursor')
            cursor.itersize = batch_size

            query = """
            SELECT axis_1, axis_2, axis_3, axis_4, axis_5, axis_6, axis_7,
                   axis_8, axis_9, axis_10, axis_11, axis_12, axis_13, axis_14, timestamp
            FROM robot_data ORDER BY id
            """
            cursor.execute(query)

            if self._dashboard_fig_num is None or not plt.fignum_exists(self._dashboard_fig_num):
                fig = plt.figure(figsize=(18, 10))
                self._dashboard_fig_num = fig.number
                self._dashboard_fig = fig

            while True:
                # Check if user closed the plot window
                if self._dashboard_fig_num is not None and not plt.fignum_exists(self._dashboard_fig_num):
                    print("\n\nPlot window closed - stopping stream...")
                    user_stopped = True
                    break

                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break

                # Convert batch to numpy array
                batch_array = np.array([[r[i] if r[i] is not None else np.nan for i in range(14)] for r in rows], dtype=np.float32)
                timestamps = [r[14] for r in rows]

                # Transfer to GPU for any computations
                gpu_batch = cp.asarray(batch_array)

                # Compute batch statistics on GPU (fast)
                batch_means = cp.nanmean(gpu_batch, axis=0)
                batch_totals = cp.nansum(gpu_batch, axis=1)

                # Transfer back to CPU for buffer storage
                batch_means_cpu = cp.asnumpy(batch_means)
                batch_totals_cpu = cp.asnumpy(batch_totals)

                # Add to buffer
                for i, row in enumerate(rows):
                    record = {
                        'Axis #1': row[0], 'Axis #2': row[1], 'Axis #3': row[2], 'Axis #4': row[3],
                        'Axis #5': row[4], 'Axis #6': row[5], 'Axis #7': row[6], 'Axis #8': row[7],
                        'Axis #9': row[8], 'Axis #10': row[9], 'Axis #11': row[10], 'Axis #12': row[11],
                        'Axis #13': row[12], 'Axis #14': row[13], 'Time': timestamps[i],
                        'total_current': float(batch_totals_cpu[i])
                    }
                    self.data_buffer.append(record)

                points_processed += len(rows)
                self.current_index += len(rows)
                batches_processed += 1

                # Visualize at intervals
                if batches_processed % visualize_every == 0:
                    record_series = pd.Series(self.data_buffer[-1])
                    self._plot_dashboard_nonblocking(record_series)

                # Process GUI events
                if batches_processed % 5 == 0:
                    plt.pause(0.001)

                # Progress update
                if batches_processed % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = points_processed / elapsed if elapsed > 0 else 0
                    print(f"\rGPU Streamed: {points_processed:,}/{total_records:,} ({points_processed/total_records*100:.1f}%) - {rate:.0f} pts/sec", end='', flush=True)

            cursor.close()
            conn.close()

        except KeyboardInterrupt:
            print("\n\nStreaming interrupted by user (Ctrl+C)")
            user_stopped = True
            try:
                cursor.close()
                conn.close()
            except:
                pass

        elapsed_time = time.time() - start_time
        rate = points_processed / elapsed_time if elapsed_time > 0 else 0

        summary = {
            'points_processed': points_processed,
            'total_records': total_records,
            'elapsed_seconds': elapsed_time,
            'points_per_second': rate,
            'buffer_size': len(self.data_buffer),
            'gpu_accelerated': True,
            'stopped_early': user_stopped
        }

        if user_stopped:
            print(f"\n\nGPU Streaming stopped by user")
        else:
            print(f"\n\nGPU Streaming complete!")
        print(f"   Points processed: {points_processed:,}/{total_records:,}")
        print(f"   Time elapsed: {elapsed_time:.2f} seconds")
        print(f"   Rate: {rate:.0f} points/second")

        return summary

    def analyze_axis_health(self):
        """
        Analyze overall health of each robot axis based on current patterns.
        
        Returns:
        --------
        dict : Health status for each axis
        """
        if len(self.data_buffer) < 50:
            print("⚠️ Need at least 50 data points for health analysis")
            return {}
        
        buffer_df = pd.DataFrame(self.data_buffer)
        health_report = {}
        
        for axis in self.axis_columns:
            if axis not in buffer_df.columns:
                continue
            
            axis_data = buffer_df[axis].dropna()
            
            if len(axis_data) < 10:
                continue
            
            # Calculate health metrics
            mean_current = axis_data.mean()
            std_current = axis_data.std()
            max_current = axis_data.max()
            cv = (std_current / mean_current * 100) if mean_current > 0 else 0  # Coefficient of variation
            
            # Determine health status
            if cv > 50:
                status = 'CRITICAL - High variability'
                color = '🔴'
            elif cv > 30:
                status = 'WARNING - Moderate variability'
                color = '🟡'
            elif max_current > 30:
                status = 'CAUTION - High current draw'
                color = '🟠'
            else:
                status = 'NORMAL'
                color = '🟢'
            
            health_report[axis] = {
                'status': status,
                'color': color,
                'mean_current': mean_current,
                'std_current': std_current,
                'max_current': max_current,
                'coefficient_of_variation': cv,
                'data_points': len(axis_data)
            }
        
        return health_report
    
    def reset(self):
        """Reset the simulator to start from beginning"""
        self.current_index = 0
        self.data_buffer = []
        # Close any existing dashboard figure to start fresh
        if self._dashboard_fig_num is not None and plt.fignum_exists(self._dashboard_fig_num):
            plt.close(self._dashboard_fig_num)
        self._dashboard_fig = None
        self._dashboard_fig_num = None
        print("🔄 Simulator reset to beginning")
    
    def get_statistics(self):
        """Get comprehensive statistics for the current data buffer"""
        if len(self.data_buffer) == 0:
            print("⚠️ No data in buffer")
            return None
        
        buffer_df = pd.DataFrame(self.data_buffer)
        
        stats = {
            'total_points': len(buffer_df),
            'time_range': {
                'start': buffer_df['Time'].min(),
                'end': buffer_df['Time'].max(),
                'duration': (buffer_df['Time'].max() - buffer_df['Time'].min())
            },
            'axes_stats': {}
        }
        
        for axis in self.axis_columns:
            if axis in buffer_df.columns:
                axis_data = buffer_df[axis].dropna()
                if len(axis_data) > 0:
                    stats['axes_stats'][axis] = {
                        'mean': axis_data.mean(),
                        'median': axis_data.median(),
                        'std': axis_data.std(),
                        'min': axis_data.min(),
                        'max': axis_data.max(),
                        'non_null_count': len(axis_data)
                    }
        
        return stats


# ============================================================================
# STANDALONE SCRIPT MODE
# Run this file directly: python StreamingSimulator.py
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("ROBOT CONTROLLER STREAMING SIMULATOR - STANDALONE MODE")
    print("=" * 80)
    print("\nPlots will open in a SEPARATE WINDOW")
    print("The simulator will stream the ENTIRE dataset from the database")
    print("One record is fetched at a time for real-time visualization\n")

    # Database configuration
    db_config = {
        'host': 'ep-polished-snow-ahx3qiod-pooler.c-3.us-east-1.aws.neon.tech',
        'database': 'neondb',
        'user': 'neondb_owner',
        'password': 'npg_JlIENr3i4AbL',
        'port': 5432,
        'sslmode': 'require'
    }

    # Path relative to src/ directory
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'RMBR4-2_export_test.csv')

    print("Configuration:")
    print(f"   Database: {db_config['host']}")
    print(f"   CSV File: {csv_path}")
    print()

    print("=" * 80)
    print("Initializing Simulator...")
    print("=" * 80)

    try:
        # Initialize simulator (loads CSV and sets up database table)
        ss = StreamingSimulator(csv_path=csv_path, db_config=db_config)

        print(f"\nTotal records in CSV: {len(ss.df):,}")

        # First, load all CSV data into the database using batch streaming
        print("\n" + "=" * 80)
        print("PHASE 1: Loading CSV data into database...")
        print("=" * 80)

        result = ss.streamBatch(
            num_points=len(ss.df),
            batch_size=1000,
            visualize_every=0,  # No visualization during loading
            plot_style='combined'
        )

        if result:
            print(f"\nDatabase loaded: {result['points_processed']:,} records")

        # Reset simulator for database streaming
        ss.reset()

        # Get total records from database
        db_count = ss.get_db_record_count()
        print(f"\nRecords in database: {db_count:,}")

        print("\n" + "=" * 80)
        print("PHASE 2: Streaming from database (single point mode)...")
        print("=" * 80)
        print("\nLook for the plot window - it will update as data streams!")
        print("Close the plot window or press Ctrl+C to stop\n")

        # Stream entire dataset from database using single-point streaming
        # This matches the notebook's logic: fetch_size=1 for true real-time streaming
        # visualize_every=1 updates the plot after every single point
        summary = ss.stream_from_db(visualize_every=1, plot_style='combined', fetch_size=1)

        print("\n" + "=" * 80)
        print("STREAMING COMPLETE")
        print("=" * 80)
        if summary:
            print(f"   Points processed: {summary['points_processed']:,}")
            print(f"   Total time: {summary['elapsed_seconds']:.2f} seconds")
            print(f"   Rate: {summary['points_per_second']:.0f} points/second")

        # Keep window open
        print("\nPlot window will stay open until you close it")
        print("Press Enter to exit or close the plot window...")

        try:
            input()
        except KeyboardInterrupt:
            pass

        plt.close('all')
        print("\nDone!")

    except FileNotFoundError:
        print(f"\nERROR: CSV file not found: {csv_path}")
        print("   Please update the csv_path variable in this script")
        sys.exit(1)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)