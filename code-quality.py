import os
import subprocess
import uuid
from datetime import datetime


def run_cmd(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.returncode, result.stdout, result.stderr


def main():
    folder_name = f"/home/bhanu/codeQuality/java_code_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    os.makedirs(folder_name, exist_ok=True)
    print(f"Created folder: {folder_name}")
    docker_container = "code-gen-ser"
    docker_path = "/root/java/generated-code/"
    print("Copying .java files from Docker container...")
    copy_cmd = [
        "docker", "cp",
        f"{docker_container}:{docker_path}",
        folder_name
    ]

    code, out, err = run_cmd(copy_cmd)
    if code != 0:
        print("Error copying files:", err)
        return

    work_dir = os.path.join(folder_name, "generated-code")
    if not os.path.isdir(work_dir):
        work_dir = folder_name

    print(f"Files copied to: {work_dir}")

    print("Deleting .java files from container...")

    delete_cmd = [
        "docker", "exec", docker_container,
        "bash", "-c", f"rm -f {docker_path}/*.java"
    ]

    _, _, del_err = run_cmd(delete_cmd)
    if del_err.strip():
        print("Warning while deleting:", del_err)

    print("All .java files deleted from container.")

    java_files = [f for f in os.listdir(work_dir) if f.endswith(".java")]
    total_files = len(java_files)

    print(f"Found {total_files} Java files to compile.")

    passed = 0
    failed = 0
    results = []

    for java_file in java_files:
        file_path = os.path.join(work_dir, java_file)
        print(f"Compiling {java_file} ...")

        code, out, err = run_cmd(["javac", file_path])

        if code == 0:
            passed += 1
        else:
            failed += 1

        results.append({
            "file": java_file,
            "status": "PASSED" if code == 0 else "FAILED",
            "error": err.strip()
        })

    quality = round((passed / total_files) * 100, 2) if total_files else 0

    print(f"Compilation complete.")
    print(f"Total: {total_files} | Passed: {passed} | Failed: {failed} | Quality: {quality}%")

    html_path = os.path.join(folder_name, "report.html")

    html = f"""
    <html>
    <head>
        <title>Code Quality Report</title>
        <style>
            body {{ font-family: Arial; margin: 40px; }}
            .summary {{ font-size: 20px; margin-bottom: 20px; }}
            .bar-container {{
                width: 80%;
                background: #eee;
                border-radius: 10px;
                height: 25px;
                margin-bottom: 20px;
            }}
            .bar {{
                height: 25px;
                border-radius: 10px;
                width: {quality}%;
                background: {'#4CAF50' if quality >= 50 else '#FF5733'};
                text-align: center;
                color: white;
                line-height: 25px;
            }}
            table {{
                border-collapse: collapse;
                width: 80%;
            }}
            th, td {{
                padding: 10px;
                border: 1px solid #ddd;
            }}
            th {{
                background: #f2f2f2;
            }}
            .pass {{ background: #d4edda; }}
            .fail {{ background: #f8d7da; }}
        </style>
    </head>
    <body>
        <h1>Java Code Quality Report</h1>

        <div class="summary">
            <p><strong>Total Files:</strong> {total_files}</p>
            <p><strong>Passed:</strong> {passed}</p>
            <p><strong>Failed:</strong> {failed}</p>
            <p><strong>Code Quality:</strong> {quality}%</p>
        </div>

        <div class="bar-container">
            <div class="bar">{quality}%</div>
        </div>

        <h2>File-wise Compilation Result</h2>
        <table>
            <tr>
                <th>Java File</th>
                <th>Status</th>
                <th>Error (if any)</th>
            </tr>
    """

    for r in results:
        cls = "pass" if r["status"] == "PASSED" else "fail"
        html += f"""
            <tr class="{cls}">
                <td>{r['file']}</td>
                <td>{r['status']}</td>
                <td><pre>{r['error']}</pre></td>
            </tr>
        """

    html += """
        </table>
    </body>
    </html>
    """

    with open(html_path, "w") as f:
        f.write(html)

    print(f"HTML report generated at: {html_path}")


if __name__ == "__main__":
    main()
