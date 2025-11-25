import threading

def start_user_check_thread(calibrator):
        """Start background thread for user recognition."""
        def check_users():
            while not calibrator._user_check_queue.empty():
                frame = calibrator._user_check_queue.get()
                result = calibrator.user_manager.find_best_match(frame)
                calibrator._user_check_result = result
        
        calibrator._user_check_thread = threading.Thread(target=check_users, daemon=True)
        calibrator._user_check_thread.start()

def stop_user_check_thread(calibrator):
        """Gracefully stop background thread."""
        if calibrator._user_check_thread and calibrator._user_check_thread.is_alive():
            # Empty the queue to allow thread to exit
            while not calibrator._user_check_queue.empty():
                try:
                    calibrator._user_check_queue.get_nowait()
                except:
                    break
            
            # Wait briefly for thread to finish
            calibrator._user_check_thread.join(timeout=0.5)