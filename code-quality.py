import os
import subprocess


def run_cmd(cmd):
    """Run a shell command and return (returncode, stdout, stderr)."""
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.returncode, result.stdout, result.stderr


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
    folder_name = f"/Users/happy/Documents/MasterThesis/results/Phase1/code-13b-gptq-run1/generated-code"

    java_files = [f for f in os.listdir(folder_name) if f.endswith(".java")]

    total_files = 100
    compile_pass = 0
    compile_fail = 0

    for java_file in java_files:
        file_path = os.path.join(folder_name, java_file)
        print(f"Compiling {file_path}...")
        code, _, err = run_cmd(["javac", file_path])
        compiled_ok = (code == 0)

        if compiled_ok:
            compile_pass += 1
        else:
            compile_fail += 1

    print(f"Files compiled: Total Files: {total_files} , {compile_pass} passed, {(total_files-compile_pass)} failed")


if __name__ == "__main__":
    main()
