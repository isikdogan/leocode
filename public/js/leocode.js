function updateProgressBar() {
    var checkboxes = document.querySelectorAll('.marked-as-solved');
    var solvedCount = Array.from(checkboxes).filter(checkbox => checkbox.checked).length;
    var progressBar = document.querySelector('#progress-bar');
    progressBar.style.width = solvedCount / checkboxes.length * 100 + '%';

    var progressText = document.querySelector('#progress-text');
    var solvedPercentage = (solvedCount / checkboxes.length * 100).toFixed(1);
    progressText.textContent = `${solvedCount} / ${checkboxes.length} solved — ${solvedPercentage}%`;
}

function downloadProgress() {
    var a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([localStorage.getItem('solvedTitles')], {type: 'application/json'}));
    a.download = 'progress.json';

    a.style.display = 'none';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

function uploadProgress(evt) {
    var file = evt.target.files[0];

    if (!file) {
        return;
    }

    var reader = new FileReader();
    reader.onload = function(e) {
        var contents = e.target.result;

        try {
            var solvedTitles = JSON.parse(contents);
            localStorage.setItem('solvedTitles', JSON.stringify(solvedTitles));

            // Update checkboxes
            var checkboxes = document.querySelectorAll('.marked-as-solved');

            checkboxes.forEach((checkbox) => {
                var title = checkbox.dataset.title;
                checkbox.checked = solvedTitles.includes(title);
                setBgColor(checkbox);
            });

            updateProgressBar();
        } catch (e) {
            alert('Error: ' + e);
        }
    };

    reader.readAsText(file);
}

function setBgColor(checkbox) {
    var detailElem = checkbox.closest('details');
    if (checkbox.checked) {
        detailElem.classList.add('solved');
    } else {
        detailElem.classList.remove('solved');
    }
}

window.addEventListener('DOMContentLoaded', (event) => {
    var checkboxes = document.querySelectorAll('.marked-as-solved');
    var solvedTitles = JSON.parse(localStorage.getItem('solvedTitles') || "[]");

    checkboxes.forEach((checkbox) => {
        var title = checkbox.dataset.title;
        checkbox.checked = solvedTitles.includes(title);
        setBgColor(checkbox);

        checkbox.addEventListener('click', () => {
            if (checkbox.checked) {
                solvedTitles.push(title);
            } else {
                var index = solvedTitles.indexOf(title);
                if (index !== -1) solvedTitles.splice(index, 1);
            }

            localStorage.setItem('solvedTitles', JSON.stringify(solvedTitles));
            setBgColor(checkbox);
            updateProgressBar();
        });
    });

    updateProgressBar();

    document.getElementById('download-button').addEventListener('click', downloadProgress);
    document.getElementById('upload-button-trigger').addEventListener('click', function() {
        document.getElementById('upload-button').click();
    });
    document.getElementById('upload-button').addEventListener('change', uploadProgress);
});