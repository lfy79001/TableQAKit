var model = 'T5-small';


function uploadFile() {
    console.log('upload file');
    var files = $('#file-selector')[0].files
    if (files.length > 0) {
        var fd = new FormData()
        fd.append('excel_file', files[0])
        $.ajax({
            method: 'POST',
            url: '/custom/upload',
            data: fd,
            processData: false,
            contentType: false,
            success: function(res) {
                console.log(res);
            }

        })
    }
}

function download_table() {
    console.log('download table');
    // TODO 下载表格
}

function select_table() {
    console.log('select table');
    // TODO 切换历史表格
}

function change_model() {
    model = $('#model-select').val();
    fetch_default_answer(dataset, split, table_idx);
}

function getAnswer(dataset, split, table_idx) {
    console.log('question:', $('#input-question').val());
    // TODO 自定义获取提问答案
}

function selectFile() {
    $('#file-selector').click();
}

$(document).ready(() => {
    $('#switchToIndex').click(function() {
        window.location.href = 'index.html';
    });
    $('#get-question-button').click(getAnswer);
    $('#upload-table').click(selectFile);
    $('#file-selector').change(uploadFile);
});