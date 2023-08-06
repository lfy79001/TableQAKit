var host = '210.75.240.136';
var port = '18890';
var url = '';
var table_name = '';

function download_table() {
    console.log('download table');
    // TODO 下载表格
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

function appendTableListHtml(name) {
    table_list_html = ' \
    <tr> \
        <td> \
            <div style="width:100px;"> \
                <b>' + name + '</b> \
            </div> \
        </td> \
        <td> \
            <div class="row"> \
                <div class="col-md-8"> \
                    <button type="button" class="select-btn btn btn-success " style="width:200px" \
                    onclick="selectTable(\'' + name + '\')"> \
                        Use This Table \
                    </button> \
                </div> \
                <div class="col-md-4"> \
                    <button type="button" class="delete-btn btn btn-danger " \
                    onclick="deleteTable(\'' + name + '\')"> \
                        Delete \
                    </button> \
                </div> \
            </div> \
        </td> \
    </tr>';
    $('#table-list').append(table_list_html);
}


function changeTableHtml(table_html) {
    $('#table-container').html(table_html);
}

function changePropertiesHtml() {
    $('#properties-container').html(properties_html);
}

function selectTable(name) {
    dataJson = { 'table_name': name };
    $.ajax({
        type: 'POST',
        url: url + '/custom/table',
        chche: false,
        async: true,
        dataType: "json",
        contentType: 'application/json',
        data: JSON.stringify(dataJson),
        success: (data) => {
            table_html = data.table_html;
            changeTableHtml(data.table_html);
            table_name = name;
            hideTable(0);
            hideTableqa(0);
        },
        error: (XMLHttpRequest, textStatus, errorThrown) => {
            alert('fetch table error!');
            console.log(XMLHttpRequest.status);
            console.log(XMLHttpRequest.readyState);
            console.log(textStatus);
        }
    });
}

// function deleteTable(index) {
//     dataJson = { 'table_name': table_list[index] };

//     $.ajax({
//         type: 'POST',
//         url: url + '/custom/remove',
//         chche: false,
//         async: true,
//         dataType: "json",
//         contentType: 'application/json',
//         data: JSON.stringify(dataJson),
//         success: (data) => {
//             if (data[success] == 'success') {
//                 table_list.splice(index);
//             } else {
//                 alert('table remove failed!');
//             }
//         },
//         error: (XMLHttpRequest, textStatus, errorThrown) => {
//             alert('fetch table error!');
//             console.log(XMLHttpRequest.status);
//             console.log(XMLHttpRequest.readyState);
//             console.log(textStatus);
//         }
//     });
// }

function submitExcelFile() {
    var files = $('#table-file-input')[0].files[0];
    var inputName = $('#table-name-input').val();
    if (inputName == null || inputName == '') {
        alert('Please input name First');
        return;
    }
    if (files == null || files.length == 0) {
        alert('Please Select File First');
        return;
    }
    var suffix = files.name.substring(files.name.lastIndexOf("."));
    var data = new FormData();
    data.append('excel_file', files, inputName+suffix);
    $.ajax({
        method: 'POST',
        url: url + '/custom/upload',
        data: data,
        processData: false,
        contentType: false,
        async: true,
        success: function(res) {
            console.log(res);
        },
        error: (XMLHttpRequest, textStatus, errorThrown) => {
            alert('fetch table error!');
            console.log(XMLHttpRequest.status);
            console.log(XMLHttpRequest.readyState);
            console.log(textStatus);
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
    download_url = url+'/custom/download?format='+$('#download-format-select').val()+'&table_name='+table_name;
    console.log(download_url);
    window.location.href(download_url);
}

function getHistory() {
    $.ajax({
        type: 'POST',
        url: url + '/session',
        chche: false,
        async: true,
        dataType: "json",
        contentType: 'application/json',
        data: JSON.stringify({"target": "custom_tables_name"}),
        success: (data) => {
            console.log(data.data);
            table_list = data.data;
            for (each in table_list) {
                appendTableListHtml(table_list[each]);
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
    hideTable(1);
    hideTableqa(1);
    getHistory();
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
    $('#download-excel').click(downloadExcel);
    $('.delete-btn').click((event) => {
        $(event.target).parent().parent().parent().parent().remove();
    });
});