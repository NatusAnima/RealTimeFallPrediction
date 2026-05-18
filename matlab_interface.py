"""
MATLAB Interface and Structural Decoupling Protocol
===================================================

This module provides the Pure Python-to-MATLAB transfer layer for real-time inference.
It operates as a lightweight TCP Socket Server that aggregates the unified synchronized buffers 
from the DAQ and serves them to MATLAB upon request.

Decoupling `master_pipeline.m` for Real-Time Execution:
-------------------------------------------------------
Currently, `master_pipeline.m` is a monolithic script built strictly for offline 
validation (LOSO cross-validation, EDF reading, and full-file preprocessing).

To transition to real-time execution in MATLAB:
1.  Extract Local Functions: 
    The `extract_features` and `normalized_acc` functions at the bottom of 
    `master_pipeline.m` must be moved into their own standalone files:
    - `extract_features.m`
    - `normalized_acc.m`
    
2.  Create a Real-Time Loop (`realtime_pipeline.m`):
    Instead of iterating over an EDF file using `for w = 1:num_windows`, MATLAB will 
    run an infinite `while` loop that connects to this Python TCP server.
    
    MATLAB Client Example:
    ```matlab
    % 1. Connect to the Python Transfer Layer
    t = tcpclient('127.0.0.1', 5005);
    
    while true
        % 2. Request the latest synchronized buffer
        write(t, uint8('GET'));
        
        % 3. Read the newline-delimited JSON response
        data = readline(t);
        buffer_dict = jsondecode(data);
        
        if strcmp(buffer_dict.status, 'waiting')
            pause(0.01);
            continue;
        end
        
        % 4. Extract arrays
        % EEG will be [256 x 4] (for 4 channels, 1000Hz)
        % ACC will be [~25 x 3] (for 3 channels, 100Hz)
        eeg_buffer = buffer_dict.eeg;
        acc_buffer = buffer_dict.acc;
        
        % 5. Call standalone decoupled functions
        % acc_norm = normalized_acc(acc_buffer');
        % if max(acc_norm) >= acc_threshold
        %     feat = extract_features(eeg_buffer, 3);
        %     pred = predict(rf_model, feat);
        % end
    end
    ```
"""

import socket
import json
import threading

class DataTransferServer:
    """
    A lightweight TCP Socket Server that runs in a background thread.
    MATLAB will act as the client, connect to this server, and request data.
    """
    def __init__(self, host='127.0.0.1', port=5005):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        
        self.latest_buffer = None
        self.lock = threading.Lock()
        self.running = True
        
        # Start server in a background thread
        self.thread = threading.Thread(target=self._server_loop, daemon=True)
        self.thread.start()
        
    def update_buffer(self, buffer_dict):
        """
        Updates the latest synchronized buffer. 
        Thread-safe method called by the DAQ Synchronizer.
        """
        with self.lock:
            self.latest_buffer = buffer_dict

    def _server_loop(self):
        print(f"[MATLAB Interface] Server listening on {self.host}:{self.port}")
        while self.running:
            try:
                self.server_socket.settimeout(1.0)
                conn, addr = self.server_socket.accept()
                with conn:
                    print(f"[MATLAB Interface] MATLAB connected from {addr}")
                    while self.running:
                        try:
                            # Wait for a request from MATLAB (e.g. 'GET')
                            data = conn.recv(1024)
                            if not data:
                                break # MATLAB disconnected
                                
                            request = data.decode('utf-8').strip()
                            if request == 'GET':
                                with self.lock:
                                    if self.latest_buffer is None:
                                        response = json.dumps({"status": "waiting"}) + "\n"
                                    else:
                                        response = json.dumps({
                                            "status": "ok",
                                            "eeg": self.latest_buffer.get("eeg", []),
                                            "acc": self.latest_buffer.get("acc", [])
                                        }) + "\n"
                                        
                                # Send newline-delimited JSON response
                                conn.sendall(response.encode('utf-8'))
                        except ConnectionResetError:
                            break
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[MATLAB Interface] Error: {e}")

    def stop(self):
        """Stops the server safely."""
        self.running = False
        try:
            self.server_socket.close()
        except:
            pass
        self.thread.join()
        print("[MATLAB Interface] Server stopped.")
