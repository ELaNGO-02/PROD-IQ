"""Simple direct test of MCP server"""
import subprocess
import json
import sys
from pathlib import Path

print("🧪 Testing MCP Server...\n")

# Use the current Python (from venv)
python_exe = sys.executable
print(f"Using Python: {python_exe}\n")

# Start server process with venv Python
server = subprocess.Popen(
    [python_exe, "mcp_server/server.py"],  # Use venv Python
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

try:
    # Send initialize request
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    }

    print("📤 Sending initialize request...")
    server.stdin.write(json.dumps(init_request) + "\n")
    server.stdin.flush()

    # Read stderr for server messages
    import threading


    def read_stderr():
        for line in server.stderr:
            print(f"📋 Server: {line.strip()}")


    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stderr_thread.start()

    # Read response
    import time

    timeout = 5
    start = time.time()

    while time.time() - start < timeout:
        if server.poll() is not None:
            print(f"❌ Server exited with code {server.returncode}")
            break

        try:
            response_line = server.stdout.readline()
            if response_line:
                print(f"✅ Response received:")
                response = json.loads(response_line)
                print(json.dumps(response, indent=2))

                # Send tools/list request
                list_request = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/list"
                }

                print("\n📤 Sending tools/list request...")
                server.stdin.write(json.dumps(list_request) + "\n")
                server.stdin.flush()

                # Read tools response
                tools_line = server.stdout.readline()
                if tools_line:
                    print(f"✅ Tools list received:")
                    tools_response = json.loads(tools_line)
                    if "result" in tools_response and "tools" in tools_response["result"]:
                        for tool in tools_response["result"]["tools"]:
                            print(f"   - {tool['name']}: {tool['description']}")

                break
        except Exception as e:
            pass

        time.sleep(0.1)
    else:
        print("❌ Timeout waiting for response")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()

finally:
    server.terminate()
    server.wait(timeout=2)
    print("\n🛑 Server stopped")
