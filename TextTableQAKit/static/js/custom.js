var model = 'T5-small';
var host = '210.75.240.136';
var port = '18889';
var url = '';

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

function submitExcelFile() {
    var files = $('#table-file-input')[0].files[0]
    if (files.length == 0) {
        alert('Please Select File First');
    }
    var data = new FormData();
    data.append('files', files);
    data.append('table_name', $('#table-name-input').val());
    $.ajax({
        method: 'POST',
        url: url + '/custom/upload',
        data: fd,
        processData: false,
        contentType: false,
        success: function(res) {
            console.log(res);
        }
    });
}

function changeSelectedExcelFile() {
    console.log('get file selected changed');
    $('#selected-file-name').val($('#table-file-input')[0].files[0].name);
}

function downloadExcel() {
    $.get(url + '/custom/download', {
        'format': '',
        'include_props': '',
        'dataset_name': '',
        'splite': '',
        'table_idx': ''
    }, (data, status) => {
        console.log(status);
        if (status === 'success') {
            var downloadUrl = window.URL.createObjectURL(data);
            window.location.href = downloadUrl;
        }
    });
}

$(document).ready(() => {
    url = 'http://' + host + ':' + port;
    $('#switchToIndex').click(function() {
        window.location.href = 'index.html';
    });
    $('#get-question-button').click(getAnswer);
    $('#upload-table').click(selectFile);
    $('#submit-excel-button').click(submitExcelFile);
    $('#select-excel-button').click(() => {
        $('#table-file-input').click()
    });
    $('#table-file-input').change(changeSelectedExcelFile);
    $('down-excel').change(downloadExcel);
});