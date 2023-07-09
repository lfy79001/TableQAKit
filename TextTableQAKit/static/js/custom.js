var model = 'T5-small';


function upload_table() {
    console.log('upload table');
    // TODO 上传表格
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

$(document).ready(function() {
    $('#switchToIndex').click(function() {
        window.location.href = 'index.html';
    });
    $('#get-question-button').click(getAnswer);
});



