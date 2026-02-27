#!/usr/bin/env python3
"""
Deploy code to VM using scp with pexpect for password authentication
"""
import sys
import pexpect
import os

VM_HOST = "10.25.33.111"
VM_USER = "test-11"
VM_PASSWORD = "Calvin@123"

def deploy_directory(local_path, remote_path):
    """Deploy a directory to VM using scp"""
    scp_cmd = f"scp -r -o StrictHostKeyChecking=no {local_path} {VM_USER}@{VM_HOST}:{remote_path}"
    
    try:
        print(f"Deploying {local_path} to {VM_USER}@{VM_HOST}:{remote_path}...")
        child = pexpect.spawn(scp_cmd, timeout=300, encoding='utf-8')
        
        # Wait for password prompt
        index = child.expect(['password:', 'Password:', pexpect.EOF, pexpect.TIMEOUT], timeout=30)
        
        if index == 0 or index == 1:
            print("Sending password...")
            child.sendline(VM_PASSWORD)
            child.expect([pexpect.EOF, pexpect.TIMEOUT], timeout=300)
            output = child.before
            print("Deployment output:")
            print(output)
            child.close()
            return child.exitstatus == 0
        else:
            print(f"Unexpected response: {child.before}")
            child.close()
            return False
            
    except pexpect.ExceptionPexpect as e:
        print(f"Pexpect error: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    # Deploy the entire phase3 directory
    local_phase3 = "/home/david/Downloads/gpu/phase3"
    remote_home = "/home/test-11"
    
    if not os.path.exists(local_phase3):
        print(f"Error: {local_phase3} does not exist")
        sys.exit(1)
    
    success = deploy_directory(local_phase3, remote_home)
    sys.exit(0 if success else 1)
