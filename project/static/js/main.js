// static/js/main.js

document.addEventListener("DOMContentLoaded", function() {
    const genderSelect = document.getElementById('gender');
    genderSelect.addEventListener('change', updateLabels);
    
    function updateLabels() {
        const gender = genderSelect.value;
        const labelsContainer = document.getElementById('labelsContainer');
        const labels = {
            'female': [
                'Big', 'Broad Shoulders', 'Feminine', 'Large Breasts', 'Long Legs',
                'Long Neck', 'Long Torso', 'Muscular', 'Pear Shaped', 'Petite', 
                'Short', 'Short Arms', 'Skinny Legs', 'Slim Waist', 'Tall'
            ],
            'male': [
                'Average', 'Big', 'Broad Shoulders', 'Delicate Build', 'Long Legs', 
                'Long Neck', 'Long Torso', 'Masculine', 'Muscular', 'Rectangular', 
                'Short', 'Short Arms', 'Skinny Arms', 'Soft Body', 'Tall'
            ]
        };
  
        labelsContainer.innerHTML = ''; // Clear existing labels
        labels[gender].forEach(function(label) {
            const div = document.createElement('div');
            div.innerHTML = `<label>${label}: </label><input type="range" name="${label}" min="0" max="5" step="0.1" value="2.5">`;
            labelsContainer.appendChild(div);
        });
    }
  
    // 페이지 로드 시 초기 레이블 설정
    updateLabels();
  });
  