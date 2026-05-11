document.addEventListener('DOMContentLoaded', () => {
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const uploadSection = document.getElementById('upload-section');
    const previewSection = document.getElementById('preview-section');
    const originalImage = document.getElementById('original-image');
    const resultImage = document.getElementById('result-image');
    const detectBtn = document.getElementById('detect-btn');
    const resetBtn = document.getElementById('reset-btn');
    const loader = document.getElementById('loader');

    let currentFile = null;

    // Handle Drag & Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.remove('dragover'), false);
    });

    dropArea.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    });

    // Handle File Input
    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            currentFile = files[0];
            
            // Preview Image
            const reader = new FileReader();
            reader.readAsDataURL(currentFile);
            reader.onloadend = function() {
                originalImage.src = reader.result;
                
                // Switch UI State
                uploadSection.classList.add('hidden');
                previewSection.classList.remove('hidden');
                resultImage.classList.add('hidden');
                resultImage.src = "";
                detectBtn.disabled = false;
            }
        }
    }

    // Handle Detection Request
    detectBtn.addEventListener('click', () => {
        if (!currentFile) return;

        // UI updates during processing
        detectBtn.disabled = true;
        loader.classList.remove('hidden');
        resultImage.classList.add('hidden');

        const formData = new FormData();
        formData.append('file', currentFile);

        fetch('/detect', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            loader.classList.add('hidden');
            if (data.success) {
                // To avoid caching issues, append timestamp
                resultImage.src = data.image_url + "?t=" + new Date().getTime();
                resultImage.classList.remove('hidden');
            } else {
                alert("Error: " + data.error);
                detectBtn.disabled = false;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            loader.classList.add('hidden');
            detectBtn.disabled = false;
            alert('An error occurred during detection.');
        });
    });

    // Handle Reset
    resetBtn.addEventListener('click', () => {
        currentFile = null;
        fileInput.value = "";
        previewSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
    });
});
