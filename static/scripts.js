function selectPart(){
	$('td').click(function(){
		$(this).toggleClass('selected');
		$('#numSamples').html($('.selected').length);
	});
}

function getSelectedParts(){
	var selectedNames = [];
	var selectedImages = $('.selected').children('img').each(function(){
		selectedNames.push($(this).attr('id'));
	});

	var names = [];
	var images = $('td').children('img').each(function(){
		names.push($(this).attr('id'));
	});

	console.dir(names);
	console.dir(selectedNames);
	var post = {};
	for(var i = 0; i < names.length; i++){
		if(selectedNames.includes(names[i])){
			post[names[i]] = true;
		}
		else{
			post[names[i]] = false;
		}
	}

	console.dir(post);
	$.ajax({
	  type: "POST",
	  contentType: 'application/json',
	  url: 'http://af423aef.ngrok.io/classify',
	  data: JSON.stringify(post),
      dataType: 'json',
	  error: function(XMLHttpRequest, textStatus, errorThrown) {
	     console.log(post);
	  }
	});
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


