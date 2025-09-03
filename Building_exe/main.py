import subprocess
import os
import sys
import webbrowser
import time
import socket

def is_port_available(port, host="localhost"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) != 0

def wait_for_port(port, host="localhost", timeout=30.0):
    start_time = time.time()
    while time.time() - start_time < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            result = sock.connect_ex((host, port))
            if result == 0:
                return True
        time.sleep(0.5)
    return False

def main():
    print("started")

    # Folder where the executable is located
    exe_dir = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, 'frozen', False) else __file__))

    # Point to the _internal folder inside dist/main/
    internal_path = os.path.join(exe_dir, "_internal")
    python_exe = os.path.join(internal_path, "embedded_python", "python.exe")
    app_py = os.path.join(internal_path, "app", "app.py")
    poppler_bin = os.path.join(internal_path, "poppler", "bin")
    user_manual_path = os.path.join(internal_path, "UserManual.pdf")
    logo_path = os.path.join(internal_path, "AIS_Logo.png")
    log_path = os.path.join(os.path.expanduser("~"), "streamlit_log.txt")

    print(f"Embedded Python path: {python_exe}")
    print(f"App path: {app_py}")
    print(f"Log file: {log_path}")

    # Log Python version
    try:
        version_check = subprocess.run(
            [python_exe, "--version"],
            capture_output=True,
            text=True
        )
        with open(log_path, "w") as log_file:
            log_file.write("Python Version Check Output:\n")
            log_file.write(version_check.stdout or version_check.stderr)
    except Exception as e:
        with open(log_path, "w") as log_file:
            log_file.write(f"Error running python.exe --version: {e}\n")
        return

    # Find a free port
    port = next(p for p in range(8561, 8600) if is_port_available(p))

    # Set environment
    env = os.environ.copy()
    env["POPLER_BIN_PATH"] = poppler_bin
    env["USER_MANUAL_PATH"] = user_manual_path
    env["LOGO_PATH"] = logo_path
    env["STREAMLIT_SERVER_PORT"] = str(port)
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["STREAMLIT_RUN_ON_SAVE"] = "false"
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
    env["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "1"

    print("Launching Streamlit subprocess...")
    with open(log_path, "a") as log_file:
        subprocess.Popen(
            [python_exe, "-m", "streamlit", "run", app_py],
            stdout=log_file,
            stderr=log_file,
            env=env
        )

    print("Waiting for Streamlit to start...")
    if wait_for_port(port, timeout=30):
        print("Streamlit is running.")
        webbrowser.open(f"http://localhost:{port}")
    else:
        print(f"ERROR: Streamlit did not start within timeout on port {port}")
        print(f"See logs in: {log_path}")

    print("done")

if __name__ == "__main__":
    main()