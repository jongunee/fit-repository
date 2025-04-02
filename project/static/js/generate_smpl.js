$(document).ready(function () {
  // "추가 신체 정보" 체크박스 체크 시, chest/waist/hips 보이게
  $('input[name="additional_info"]').change(function () {
    if ($(this).is(":checked")) {
      $("#additionalInfoSection").show();
    } else {
      $("#additionalInfoSection").hide();
    }
  });

  // "언어적 속성 정보" 체크 시, label rating 슬라이더 보이게
  $('input[name="linguistic_info"]').change(function () {
    if ($(this).is(":checked")) {
      $("#linguisticInfoSection").show();
    } else {
      $("#linguisticInfoSection").hide();
    }
  });

  // 폼 제출 이벤트 처리
  $(document).on("submit", "#generateSmplForm", function (e) {
    e.preventDefault();
    $.ajax({
      type: "POST",
      url: "/generate_smpl", // 올바른 엔드포인트로 설정
      data: $(this).serialize(),
      success: function (response) {
        // 성공적으로 생성되었을 때 처리
        // 예를 들어, 페이지를 새로고침하거나 성공 메시지를 표시
        window.location.reload();
      },
      error: function (xhr, status, error) {
        console.error("Error:", error);
        alert("An error occurred while generating the SMPL model.");
      },
    });
  });
});
