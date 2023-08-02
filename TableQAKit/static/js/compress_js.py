import os
from jsmin import jsmin

def concat_and_compress_files(file_list):
    # 合并文件内容
    combined_content = ""
    for file_path in file_list:
        with open(file_path, "r") as f:
            combined_content += f.read() + "\n"

    return combined_content

def main():

    js_files = ["dashboard.js", "data-table.js", "dataTables.bootstrap4.js", "file-upload.js",
                 "hoverable-collapse.js", "jquery.cookie.js", "jquery.dataTables.js", "jquery.js", 
                 "js.dataTables.bootstrap4.js", "js.jquery.dataTables.js", "off-canvas.js", "template.js"]

    # 合并 JavaScript 文件并压缩
    compressed_js = concat_and_compress_files(js_files)
    with open("bundle.js", "w") as f:
        f.write(jsmin(compressed_js))

if __name__ == "__main__":
    main()