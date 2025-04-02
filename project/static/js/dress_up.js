// /static/js/submit_fitting_room.js

$(document).on('submit', '#fittingRoomForm', function(e) {
    e.preventDefault();
    $.ajax({
        type: 'POST',
        url: '/fitting_room',  // 올바른 엔드포인트로 설정
        data: $(this).serialize(),
        success: function(response) {
            // 페이지 내용을 동적으로 업데이트
            $('body').html(response);
        },
        error: function(xhr, status, error) {
            console.error("Error:", error);
            alert("An error occurred while processing the fitting room.");
        }
    });
  });
  