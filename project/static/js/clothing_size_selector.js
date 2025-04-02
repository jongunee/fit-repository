// /static/js/clothing_size_selector.js

document.addEventListener('DOMContentLoaded', function() {
  const serialNumberRadios = document.getElementsByName('serial_number');
  const sizeSelect = document.getElementById('clothing_size');

  function updateSizes() {
      sizeSelect.innerHTML = '';
      const selectedSerialNumber = document.querySelector('input[name="serial_number"]:checked').value;
      const selectedCloth = clothes.find(cloth => cloth.serial_number === selectedSerialNumber);
      if (selectedCloth) {
          selectedCloth.sizes.forEach(function(size) {
              const option = document.createElement('option');
              option.value = size;
              option.text = size;
              sizeSelect.appendChild(option);
          });
      }
  }

  // 라디오 버튼에 이벤트 리스너 추가
  serialNumberRadios.forEach(function(radio) {
      radio.addEventListener('change', updateSizes);
  });

  // 페이지 로드 시 사이즈 초기화
  if (serialNumberRadios.length > 0) {
      serialNumberRadios[0].checked = true;
      updateSizes();
  }
});
