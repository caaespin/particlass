function selectPart(){
	$('td').click(function(){
		$(this).addClass('selected');
	});
}

function getSelectedParts(){
	
}

function bindScroll(){
		console.log('bound');
	//$(window).scroll(function (e) {
	    pos = 365;
	    if ($(window).scrollTop() > pos) {
	        $('#activeContent').css({
	            position: 'relative',
	            top: 0,
	            bottom: 'auto'
	        });
	    } else {
	        $('#activeContent').css({
	            position: 'fixed',
	            top: 'auto',
	            bottom: 0
	        });
	    }
	//});
}

function mobileThings(){
  if(window.innerHeight > window.innerWidth){
    $('.arrow').attr('src', 'assets/arrowM.svg');
    $(window).on('scroll', bindScroll);
  }
  else{
	console.log('unbind');
    $(window).off('scroll');
    
    $('#activeContent').css({
        position: 'relative',
        top: 0,
        bottom: 'auto'
    });
  }
}