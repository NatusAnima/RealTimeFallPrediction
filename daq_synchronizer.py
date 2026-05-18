import time
import threading
import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock, StreamInlet, resolve_byprop
import tkinter as tk
import csv
from datetime import datetime
import pyedflib
from matlab_interface import DataTransferServer

def create_brainbit_outlet():
    """
    Placeholder function to set up an LSL Outlet for a 4-channel EEG stream (simulating BrainBit).
    Assumes 1000 Hz sampling rate.
    In practice, the actual BrainBit Python SDK would push data to this outlet.
    """
    info = StreamInfo('BrainBit', 'EEG', 4, 1000, 'float32', 'brainbit_mock_1234')
    outlet = StreamOutlet(info)
    print("[LSL] BrainBit EEG Outlet created.")
    return outlet

class DummyACCNode:
    """
    Creates an LSL Outlet pushing 3-axis accelerometer data (simulated 100Hz gravity/noise).
    """
    def __init__(self):
        self.info = StreamInfo('DummyACC', 'ACC', 3, 100, 'float32', 'dummy_acc_1234')
        self.outlet = StreamOutlet(self.info)
        self.running = False
        self.thread = threading.Thread(target=self._push_data)
        
    def start(self):
        self.running = True
        self.thread.start()
        print("[LSL] Dummy ACC Node started.")
        
    def stop(self):
        self.running = False
        self.thread.join()
        
    def _push_data(self):
        # Base gravity vector (Z=9.8, X=0, Y=0) + some noise
        base_g = np.array([0.0, 0.0, 9.8])
        while self.running:
            # Pushing in chunks of 10 to ensure stable 100Hz without relying on sleep precision
            noise = np.random.normal(0, 0.1, (10, 3))
            samples = base_g + noise
            self.outlet.push_chunk(samples.tolist())
            time.sleep(0.1) # 10 chunks/sec * 10 = 100Hz

class EventMarkerGUI:
    """
    Calibration GUI for Event Tagging.
    """
    def __init__(self, master, marker_outlet, on_close_callback):
        self.master = master
        self.marker_outlet = marker_outlet
        self.on_close_callback = on_close_callback
        master.title("Calibration GUI - Event Marker")
        master.geometry("350x250")
        
        lbl = tk.Label(master, text="Tag Real-Time Events", font=("Arial", 14, "bold"))
        lbl.pack(pady=10)

        self.btn_pred = tk.Button(master, text="Predicted Fall (Class 0)", bg="lightblue", font=("Arial", 12), command=self.mark_pred)
        self.btn_pred.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)
        
        self.btn_unexp = tk.Button(master, text="Unexpected Fall (Class 1)", bg="salmon", font=("Arial", 12), command=self.mark_unexp)
        self.btn_unexp.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)
        
        master.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def mark_pred(self):
        print(f"[GUI] Marker Pushed: Predicted Fall (Class 0)")
        self.marker_outlet.push_sample(["Class 0: Predicted Fall"])

    def mark_unexp(self):
        print(f"[GUI] Marker Pushed: Unexpected Fall (Class 1)")
        self.marker_outlet.push_sample(["Class 1: Unexpected Fall"])
        
    def on_close(self):
        self.on_close_callback()

class Synchronizer:
    """
    Aligns LSL streams and provides a unified 256ms buffer to the TCP Server.
    Also handles recording data and dumping to .edf/.csv.
    """
    def __init__(self, server):
        self.server = server
        self.running = False
        self.thread = threading.Thread(target=self._sync_loop)
        
        # Data storage for saving at the end
        self.eeg_full_data = []
        self.acc_full_data = []
        self.markers = []
        
    def start(self):
        self.running = True
        self.thread.start()
        print("[Sync] Synchronizer started.")
        
    def stop(self):
        self.running = False
        self.thread.join()
        
    def _sync_loop(self):
        print("[Sync] Resolving LSL streams...")
        # Using timeout allows the script to fail gracefully if streams aren't running
        eeg_streams = resolve_byprop('type', 'EEG', timeout=5)
        acc_streams = resolve_byprop('type', 'ACC', timeout=5)
        marker_streams = resolve_byprop('type', 'Markers', timeout=5)
        
        if not eeg_streams or not acc_streams or not marker_streams:
            print("[Sync] Error: Could not resolve all required streams.")
            self.running = False
            return
            
        eeg_inlet = StreamInlet(eeg_streams[0])
        acc_inlet = StreamInlet(acc_streams[0])
        marker_inlet = StreamInlet(marker_streams[0])
        
        print("[Sync] All streams connected. Starting synchronization loop.")
        
        # Buffer dimensions
        eeg_window_samples = 256 # At 1000 Hz, 256ms = 256 samples
        acc_window_samples = 25  # At 100 Hz, 256ms = ~25 samples
        
        eeg_buffer = []
        acc_buffer = []
        
        while self.running:
            # Pull EEG data
            try:
                eeg_chunk, eeg_timestamps = eeg_inlet.pull_chunk(timeout=0.0)
                if eeg_chunk:
                    eeg_buffer.extend(eeg_chunk)
                    self.eeg_full_data.extend(eeg_chunk)
            except Exception:
                pass
                
            # Pull ACC data
            try:
                acc_chunk, acc_timestamps = acc_inlet.pull_chunk(timeout=0.0)
                if acc_chunk:
                    acc_buffer.extend(acc_chunk)
                    self.acc_full_data.extend(acc_chunk)
            except Exception:
                pass
                
            # Pull Markers
            try:
                marker, marker_ts = marker_inlet.pull_sample(timeout=0.0)
                if marker:
                    timestamp_str = datetime.now().strftime('%H:%M:%S.%f')
                    self.markers.append([timestamp_str, marker[0]])
            except Exception:
                pass
                
            # Keep rolling buffer at exactly the target window size
            if len(eeg_buffer) > eeg_window_samples:
                eeg_buffer = eeg_buffer[-eeg_window_samples:]
            
            if len(acc_buffer) > acc_window_samples:
                acc_buffer = acc_buffer[-acc_window_samples:]
                
            # Only push if we have a full 256ms buffer for both sensors
            if len(eeg_buffer) == eeg_window_samples and len(acc_buffer) == acc_window_samples:
                buffer_dict = {
                    "eeg": eeg_buffer,
                    "acc": acc_buffer
                }
                # Forward to MATLAB interface server
                self.server.update_buffer(buffer_dict)
                
            time.sleep(0.01) # Sleep slightly to prevent maxing out CPU (10ms)

    def save_data(self, edf_filename="session_data.edf", csv_filename="session_markers.csv"):
        print(f"[Record] Saving data to {edf_filename} and {csv_filename}...")
        
        # 1. Save Markers to CSV
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Event'])
            writer.writerows(self.markers)
            
        # 2. Save Data to EDF
        if not self.eeg_full_data:
            print("[Record] No EEG data to save.")
            return
            
        try:
            # Convert to numpy arrays, shape: (channels, samples)
            eeg_arr = np.array(self.eeg_full_data).T 
            acc_arr = np.array(self.acc_full_data).T
            
            channel_info = []
            
            # Setup EEG Headers
            for i in range(eeg_arr.shape[0]):
                channel_info.append({
                    'label': f'EEG_{i+1}', 
                    'dimension': 'uV', 
                    'sample_rate': 1000, 
                    'physical_max': 500, 
                    'physical_min': -500, 
                    'digital_max': 32767, 
                    'digital_min': -32768, 
                    'transducer': 'BrainBit', 
                    'prefilter': ''
                })
            
            # Setup ACC Headers
            for i, axis in enumerate(['X', 'Y', 'Z']):
                channel_info.append({
                    'label': f'ACC_{axis}', 
                    'dimension': 'g', 
                    'sample_rate': 100, 
                    'physical_max': 16, 
                    'physical_min': -16, 
                    'digital_max': 32767, 
                    'digital_min': -32768, 
                    'transducer': 'Accelerometer', 
                    'prefilter': ''
                })
            
            signals = []
            for i in range(eeg_arr.shape[0]):
                signals.append(eeg_arr[i])
            for i in range(acc_arr.shape[0]):
                signals.append(acc_arr[i])
                
            # pyedflib highlevel requires signals to be a list of numpy arrays
            import pyedflib.highlevel
            pyedflib.highlevel.write_edf(edf_filename, signals, channel_info)
            print("[Record] EDF saved successfully.")
            
        except Exception as e:
            print(f"[Record] Failed to save EDF: {e}")


def main():
    print("Initializing DAQ Architecture...")
    # 1. Create LSL Outlets
    eeg_outlet = create_brainbit_outlet()
    acc_node = DummyACCNode()
    acc_node.start()
    
    # Marker Outlet
    marker_info = StreamInfo('CalibrationMarkers', 'Markers', 1, 0, 'string', 'markers_1234')
    marker_outlet = StreamOutlet(marker_info)
    print("[LSL] Calibration Marker Outlet created.")
    
    # 2. Start Transfer Server (MATLAB Interface)
    server = DataTransferServer(port=5005)
    
    # 3. Start Synchronizer
    synchronizer = Synchronizer(server)
    synchronizer.start()
    
    # Mocking BrainBit Data Injection (Since it's a placeholder)
    def push_eeg_mock():
        while synchronizer.running:
            # Push in chunks of 10 to ensure stable 1000Hz
            samples = np.random.normal(0, 5, (10, 4))
            eeg_outlet.push_chunk(samples.tolist())
            time.sleep(0.01) # 100 chunks/sec * 10 = 1000Hz
            
    eeg_mock_thread = threading.Thread(target=push_eeg_mock, daemon=True)
    eeg_mock_thread.start()
    
    # 4. Create and run Tkinter GUI
    root = tk.Tk()
    
    def on_closing():
        print("Shutting down... Please wait for recordings to save.")
        acc_node.stop()
        synchronizer.stop()
        server.stop()
        root.destroy()
        
        # Save data
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        synchronizer.save_data(f"session_data_{timestamp_str}.edf", f"session_markers_{timestamp_str}.csv")
        
    app = EventMarkerGUI(root, marker_outlet, on_closing)
    root.mainloop()

if __name__ == '__main__':
    main()
