import os
import subprocess
import uuid
from datetime import datetime
import json

# Path to your HumanEval-style test JSON
TEST_FILE = "tests.json"


def run_cmd(cmd):
    """Run a shell command and return (returncode, stdout, stderr)."""
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.returncode, result.stdout, result.stderr


def load_tests():
    """Load test cases from JSON file."""
    if not os.path.exists(TEST_FILE):
        return []
    with open(TEST_FILE, "r") as f:
        return json.load(f)


def run_java_test(class_name, test):
    """
    Run a compiled Java class with a test input.
    Java class must print only the answer.
    """
    input_str = ",".join(map(str, test["input"]))
    cmd = ["java", class_name, input_str]
    code, out, err = run_cmd(cmd)

    if code != 0:
        return False, f"Runtime Error: {err}"

    output = out.strip()
    try:
        val = int(output)
    except:
        return False, f"Invalid output: {output}"

    return val == test["expected"], f"expected={test['expected']} got={val}"


def main():
    # ----------------------------
    # Setup folder and docker copy
    # ----------------------------
    folder_name = f"/home/bhanu/codeQuality/java_code_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    os.makedirs(folder_name, exist_ok=True)
    docker_container = "code-gen-ser"
    docker_path = "/root/java/generated-code/"

    # Copy .java files from Docker container
    print("Copying Java files from Docker container...")
    code, out, err = run_cmd(["docker", "cp", f"{docker_container}:{docker_path}", folder_name])
    if code != 0:
        print("Error copying files:", err)
        return

    work_dir = os.path.join(folder_name, "generated-code")
    if not os.path.isdir(work_dir):
        work_dir = folder_name

    # Delete .java files from container
    run_cmd(["docker", "exec", docker_container, "bash", "-c", f"rm -f {docker_path}/*.java"])

    # ----------------------------
    # Prepare Java files and tests
    # ----------------------------
    java_files = [f for f in os.listdir(work_dir) if f.endswith(".java")]
    total_files = len(java_files)

    compile_pass = 0
    compile_fail = 0
    test_pass = 0
    test_fail = 0

    results = []
    tests = load_tests()

    for java_file in java_files:
        file_path = os.path.join(work_dir, java_file)

        # ---------------------------
        # Compile the Java file
        # ---------------------------
        print(f"Compiling {java_file}...")
        code, _, err = run_cmd(["javac", file_path])
        compiled_ok = (code == 0)

        if compiled_ok:
            compile_pass += 1
        else:
            compile_fail += 1

        file_result = {
            "file": java_file,
            "compile_status": "PASS" if compiled_ok else "FAIL",
            "compile_error": err.strip(),
            "tests": []
        }

        # ------------------------------------
        # Run tests only if compilation passed
        # ------------------------------------
        if compiled_ok:
            class_name = java_file.replace(".java", "")

            for t in tests:
                if t["functionName"].lower() == class_name.lower():
                    for test_case in t["tests"]:
                        ok, msg = run_java_test(class_name, test_case)
                        if ok:
                            test_pass += 1
                        else:
                            test_fail += 1

                        file_result["tests"].append({
                            "input": test_case["input"],
                            "expected": test_case["expected"],
                            "status": "PASS" if ok else "FAIL",
                            "message": msg
                        })
        else:
            # Increment test_fail automatically if compilation failed
            test_fail += 1
            file_result["tests"].append({
                "input": "-",
                "expected": "-",
                "status": "FAIL",
                "message": "Test not run due to compilation failure"
            })

        results.append(file_result)

    total_tests = test_pass + test_fail

    # ----------------------------
    # Generate HTML report
    # ----------------------------
    html_path = os.path.join(folder_name, "report.html")

    html = f"""
    <html>
    <head>
        <title>Java Compilation & Test Report</title>
        <style>
            body {{ font-family: Arial; margin: 40px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; }}
            th {{ background: #f2f2f2; }}
            .pass {{ background: #d4edda; }}
            .fail {{ background: #f8d7da; }}
        </style>
    </head>
    <body>
        <h1>Java Compilation & Test Report</h1>

        <h2>Summary</h2>
        <ul>
            <li><strong>Total Files:</strong> {total_files}</li>
            <li><strong>Compiled Successfully:</strong> {compile_pass}</li>
            <li><strong>Compilation Failed:</strong> {compile_fail}</li>
            <li><strong>Total Test Cases:</strong> {total_tests}</li>
            <li><strong>Test Cases Passed:</strong> {test_pass}</li>
            <li><strong>Test Cases Failed:</strong> {test_fail}</li>
        </ul>

        <h2>File Results</h2>
        <table>
            <tr>
                <th>File</th>
                <th>Compile Status</th>
                <th>Test Results</th>
            </tr>
    """

    for r in results:
        cls = "pass" if r["compile_status"] == "PASS" else "fail"
        tests_html = "<br>".join([
            f"{t['status']} | input={t['input']} | expected={t['expected']} | {t['message']}"
            for t in r["tests"]
        ])
        html += f"""
        <tr class="{cls}">
            <td>{r['file']}</td>
            <td>{r['compile_status']}</td>
            <td><pre>{tests_html if tests_html else '-'}</pre></td>
        </tr>
        """

    html += """
        </table>
    </body>
    </html>
    """

    with open(html_path, "w") as f:
        f.write(html)

    print(f"\nReport generated at {html_path}")
    print(f"Files compiled: {compile_pass} passed, {compile_fail} failed")
    print(f"Test cases: {test_pass} passed, {test_fail} failed")


if __name__ == "__main__":
    main()
