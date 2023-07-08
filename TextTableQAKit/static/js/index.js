var split = 'dev';
var dataset = 'wikisql0';
var model = 'T5-small';
var table_idx = 0;
var total_examples = 1;

function mod(n, m) {
    return ((n % m) + m) % m;
}

function change_split() {
    split = $('#split-select').val();
    table_idx = 0;
    fetch_table(dataset, split, table_idx);
}

function change_dataset() {
    dataset = $('#dataset-select').val();
    table_idx = 0;
    fetch_dataset(dataset, split, table_idx);
}

function change_model() {
    model = $('#model-select').val();
    fetch_default_answer(dataset, split, table_idx);
}


function fetch_table(dataset, split, table_idx) {
    console.log('fetch_table{dataset:', dataset, 'split:', split, 'table_idx:', table_idx, '}');
    // TODO 拉取index页面中的表格
}

function fetch_pic(dataset, split, table_idx) {
    console.log('fetch_table{dataset:', dataset, 'split:', split, 'table_idx:', table_idx, '}');
    // TODO 拉取index页面数据集的图片
}

function fetch_text(dataset, split, table_idx) {
    console.log('fetch_table{dataset:', dataset, 'split:', split, 'table_idx:', table_idx, '}');
    // TODO 拉取index页面数据集的文本
}

function fetch_default_answer(dataset, split, table_idx) {
    console.log('fetch_default_answer{dataset:', dataset, 'split:', split, 'table_idx:', table_idx, '}');
    // TODO 拉取index页面数据集的问题回答
}

function download(dataset, split) {
    console.log('download_dataset{dataset:', dataset, 'split:', split, '}');
    // TODO 下载数据集
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

    fetch_table(dataset, split, table_idx);
    $("#page-input").val(table_idx);
}

function getAnswer(dataset, split, table_idx) {
    console.log('question:', $('#input-question').val());
    // TODO 自定义获取提问答案
}

$(document).ready(() => {
    $('#dataset-select').val(dataset);
    $('#splite-select').val(split);
    $('#model-select').val(model);
    $("#dataset-select").change(change_dataset);
    $("#split-select").change(change_split);
    $('#model-select').change(change_model);
    $("#total-examples").html(total_examples - 1);
    $('#switchToCustomMode').click(() => {
        window.location.href = 'custom_mode.html';
    });
    $('#page-input').keypress(function(event) {
        if (event.keyCode == 13) {
            gotobtn();
        }
    });
    $('#get-question-button').click(getAnswer);
});