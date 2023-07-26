var split = 'dev';
var dataset = 'finqa';
var default_model = 'T5-small';
var model = 'T5-small';
var table_idx = 0;
var total_examples = 1;
var default_result = {};
var default_question = '';
var properties_html = '';
var table_html = '';
var pictures = {};
var text = {};
var host = '210.75.240.136';
var port = '18889';
var url = '';

function mod(n, m) {
    return ((n % m) + m) % m;
}

function splitChanged() {
    split = $('#split-select').val();
    table_idx = 0;
    getData()
}

function datasetChanged() {
    dataset = $('#dataset-select').val();
    table_idx = 0;
    getData();
}

function defaultModelChanged() {
    default_model = $('#default-model-select').val();
    if (default_model in default_result) {
        $('#default-model-answer').html(default_model[default_model]);
    }
}

function changeModel() {
    model = $('#model-select').val();
}

function changeDefaultQuestionHtml() {
    $('#default-question').html(question);
}

function changeTableHtml() {
    $('#table-container').html(table_html);
}

function changePictureHtml() {
    var pictures_html = ''
    for (let pic in pictures) {
        pictures_html += '<div class="col-sm-6 col-md-4 col-lg-3"><img src="../static/img/' + pic + '" style="max-width: 100%; height: auto;"></div>'
    }
    $('#image-container').html(pictures_html);
}

function changeDefaultResult() {
    $('#default-model-answer').html(JSON.stringify(default_result));
}

function changePropertiesHtml() {
    $('#properties-container').html(properties_html);
}

function changeTextHtml() {
    var text_html = '';
    for (var index in text) {
        text_html += '<tr><td>' + index + '</td><td>' + text[index] + '</td></tr>';
    }
    $("#text-container").html(text_html);
}

function changeTotalexamples() {
    $("#total-examples").html(total_examples - 1);
}

function getData() {
    var dataJson = { 'dataset_name': dataset, 'split': split, 'table_idx': table_idx };
    $.ajax({
        type: 'POST',
        url: url + '/default/table',
        chche: false,
        async: false,
        dataType: "json",
        contentType: 'application/json',
        data: JSON.stringify(dataJson),
        success: (data) => {
            console.log(data);
            total_examples = data.table_cnt;
            changeTotalexamples();
            default_result = data.generated_results;
            changeDefaultResult();
            dataset_info = data.dataset_info;
            default_question = data.table_question;
            changeDefaultQuestionHtml();
            properties_html = data.properties_html;
            changePropertiesHtml();
            table_html = data.table_html;
            changeTableHtml();
            pictures = data.pictures;
            changePictureHtml();
            text = data.text;
            changeTextHtml();

        },
        error: (XMLHttpRequest, textStatus, errorThrown) => {
            alert('fetch table error!');
            console.log(XMLHttpRequest.status);
            console.log(XMLHttpRequest.readyState);
            console.log(textStatus);
        }
    });
}

function generateAnswer() {
    // TODO pipeline
}

function nextbtn() {
    gotopage(table_idx + 1);
}

function prevbtn() {
    gotopage(table_idx - 1);
}

function startbtn() {
    gotopage(0);
}

function endbtn() {
    gotopage(total_examples - 1);
}

function gotobtn() {
    var n = $("#page-input").val();
    gotopage(n);
}

function gotopage(page) {
    table_idx = page;
    table_idx = mod(table_idx, total_examples);
    $("#page-input").val(table_idx);
    getData();
}


function downloadTable() {
    var dataJson = {
        'format': $('#download-format-select').val(),
        'include_props': ($('#includes-properties').val() == 'on') ? true : false,
        'dataset_name': dataset,
        'split': split,
        'table_idx': table_idx
    };
    console.log(dataJson);
    $.ajax({
        type: 'POST',
        url: url + '/default/download',
        chche: false,
        async: false,
        dataType: 'binary',
        contentType: 'application/json',
        xhrFields: {
            'responseType': 'blob'
        },
        data: JSON.stringify(dataJson),
        success: (data, status, xhr) => {
            console.log(xhr);
            const download_URL = (window.URL || window.webkitURL).createObjectURL(result);
            const a_link = document.createElement('a');
            a_link.href = download_URL;
            // 利用了a标签的download属性,指定文件名称
            a_link.download = 'file.' + dataJson['format'];
            document.body.appendChild(a_link);
            a_link.click();

            setTimeout(function() {
                // 移除内存中的临时文件路径和为下载而创建的a标签
                URL.revokeObjectURL(download_URL);
                a_link.remove();
            }, 10000);
        },
        error: (XMLHttpRequest, textStatus, errorThrown) => {
            alert('fetch file error!');
            console.log(XMLHttpRequest.status);
            console.log(XMLHttpRequest.readyState);
            console.log(textStatus);
        }
    });
}

function initPage() {
    table_cnt = 0;
    total_examples = 1;
    getData();
    // changeDefaultQuestionHtml();
    // changeTableHtml();
    // changePropertiesHtml();
    // changeTextHtml();
    // changeTotalexamples();
}

$(document).ready(() => {
    url = "http://" + host + ':' + port;
    initPage();
    $('#dataset-select').val(dataset);
    $('#splite-select').val(split);
    $('#default-model-select').val(default_model);
    $('#model-select').val(model);
    $("#dataset-select").change(datasetChanged);
    $("#split-select").change(splitChanged);
    $('#default-model-select').change(defaultModelChanged);
    $('#model-select').change(changeModel);
    $('#switchToCustomMode').click(() => {
        window.location.href = 'custom_mode.html';
    });
    $('#page-input').keypress(function(event) {
        if (event.keyCode == 13) {
            gotobtn();
        }
    });
    $('#download-table').click(downloadTable);
});