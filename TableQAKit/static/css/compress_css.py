import os
from csscompressor import compress as compress_css
from jsmin import jsmin

def concat_and_compress_files(file_list):
    # 合并文件内容
    combined_content = ""
    for file_path in file_list:
        with open(file_path, "r") as f:
            combined_content += f.read() + "\n"

    return combined_content

def main():
    css_files = ["dataTables.bootstrap4.css", "style.css", "vendor.bundle.base.css"]

    # 合并 CSS 文件并压缩
    compressed_css = concat_and_compress_files(css_files)
    with open("bundle.css", "w") as f:
        f.write(compress_css(compressed_css))


if __name__ == "__main__":
    main()