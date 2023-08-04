var table_html = '';
var properties_html = '';
var table_list = [];
var model = 'T5-small';
var host = '210.75.240.136';
var port = '18890';
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

function changeTableListHtml() {
    var table_list_html = '<tr><th><div style="width:100px;">Table Name</div></th><th><div>Action</div></th></tr>'
    for (var index = 0; index < table_list.length; index) {
        table_list_html += '<tr><td><div style="width:100px;"><b>' + table_list[index] + '</b></div></td><td><div class="row"><div class="col-md-8"><button type="button" class="btn btn-success " style="width:200px" onclick="selectTable(' + index + ')">Use This Table</button></div><div class="col-md-4"><button type="button" class="btn btn-danger " onclick="deleteTable(' + index + ')">Delete</button></div></div></td></tr>'
    }
    $('#table-list').html(table_list_html);
}

function changeTableHtml() {
    $('#table-container').html(table_html);
}

function changePropertiesHtml() {
    $('#properties-container').html(properties_html);
}

function selectTable(index) {
    dataJson = { 'table_name': table_list[index] };

    $.ajax({
        type: 'POST',
        url: url + '/custom/table',
        chche: false,
        async: true,
        dataType: "json",
        contentType: 'application/json',
        data: JSON.stringify(dataJson),
        success: (data) => {
            if (data[success] == 'success') {
                table_html = data.table_html;
                changeTableHtml();
                properties_html = data.properties_html;
                changePropertiesHtml();
            } else {
                alert('fetch table failed!');
            }
        },
        error: (XMLHttpRequest, textStatus, errorThrown) => {
            alert('fetch table error!');
            console.log(XMLHttpRequest.status);
            console.log(XMLHttpRequest.readyState);
            console.log(textStatus);
        }
    });
}

function deleteTable(index) {
    dataJson = { 'table_name': table_list[index] };

    $.ajax({
        type: 'POST',
        url: url + '/custom/remove',
        chche: false,
        async: true,
        dataType: "json",
        contentType: 'application/json',
        data: JSON.stringify(dataJson),
        success: (data) => {
            if (data[success] == 'success') {
                table_list.splice(index);
                changeTableListHtml();
            } else {
                alert('table remove failed!');
            }
        },
        error: (XMLHttpRequest, textStatus, errorThrown) => {
            alert('fetch table error!');
            console.log(XMLHttpRequest.status);
            console.log(XMLHttpRequest.readyState);
            console.log(textStatus);
        }
    });
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
        async: true,
        success: function(res) {
            console.log(res);
        }
    });
}

function getTable() {
    var dataJson = { 'table_name': tablename };
    $.ajax({
        type: 'POST',
        url: url + '/custom/table',
        chche: false,
        async: false,
        dataType: "json",
        contentType: 'application/json',
        data: JSON.stringify(dataJson),
        success: (data) => {
            properties_html = data.properties_html;
            table_html = data.table_html;
            changeTableHtml();
            changePropertiesHtml();
        },
        error: (XMLHttpRequest, textStatus, errorThrown) => {
            alert('fetch table error!');
            console.log(XMLHttpRequest.status);
            console.log(XMLHttpRequest.readyState);
            console.log(textStatus);
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

function downloadExcelExample() {
    window.location.href = url + '/custom/example';
}

function hideTable(val) {
    if (val == 1) {
        $("#table").hide();
    } else {
        $("#table").show();
    }
}

function hideTableqa(val) {
    if (val == 1) {
        $("#tableqa").hide();
    } else {
        $("#tableqa").show();
    }
}

function init() {
    changeTableListHtml();
    hideTable(1);
    hideTableqa(1);
}

$(document).ready(() => {
    url = 'http://' + host + ':' + port;
    init();
    $('#switchToIndex').click(() => {
        window.location.href = 'index.html';
    });
    $('#get-question-button').click(getAnswer);
    $('#upload-table').click(selectFile);
    $('#submit-excel-button').click(submitExcelFile);
    $('#select-excel-button').click(() => {
        $('#table-file-input').click();
    });
    $('#table-file-input').change(changeSelectedExcelFile);
    $('#down-excel').change(downloadExcel);
});