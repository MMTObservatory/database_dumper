/*!
    * Start Bootstrap - Grayscale v6.0.2 (https://startbootstrap.com/themes/grayscale)
    * Copyright 2013-2020 Start Bootstrap
    * Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-grayscale/blob/master/LICENSE)
    
    (function ($) {
    "use strict"; // Start of use strict

    // Smooth scrolling using jQuery easing
    $('a.js-scroll-trigger[href*="#"]:not([href="#"])').click(function () {
        if (
            location.pathname.replace(/^\//, "") ==
                this.pathname.replace(/^\//, "") &&
            location.hostname == this.hostname
        ) {
            var target = $(this.hash);
            target = target.length
                ? target
                : $("[name=" + this.hash.slice(1) + "]");
            if (target.length) {
                $("html, body").animate(
                    {
                        scrollTop: target.offset().top - 70,
                    },
                    1000,
                    "easeInOutExpo"
                );
                return false;
            }
        }
    });

    // Closes responsive menu when a scroll trigger link is clicked
    $(".js-scroll-trigger").click(function () {
        $(".navbar-collapse").collapse("hide");
    });

    // Activate scrollspy to add active class to navbar items on scroll
    $("body").scrollspy({
        target: "#mainNav",
        offset: 100,
    });

    // Collapse Navbar
    var navbarCollapse = function () {
        if ($("#mainNav").offset().top > 100) {
            $("#mainNav").addClass("navbar-shrink");
        } else {
            $("#mainNav").removeClass("navbar-shrink");
        }
    };
    // Collapse now if page is not at top
    navbarCollapse();
    // Collapse the navbar when page is scrolled
    $(window).scroll(navbarCollapse);
})(Query); // End of use strict
*/

$(function()
{
     $('input[name="startdate"]').datepicker({
        opens: 'left'
     },
     function(start, end, label) {});

    $('input[name="enddate"]').datepicker
    ({
        opens: 'left'
    },
    function(start, end, label) {});
		
    end = new Date();
		start = new Date( end-7*24*3600*1000 );
    $('input[name="startdate"]').val((start.getUTCMonth()+1)+"/"+start.getUTCDate()+"/"+start.getUTCFullYear())
    $('input[name="enddate"]').val((end.getUTCMonth()+1)+"/"+end.getUTCDate()+"/"+end.getUTCFullYear())

     $( "#slider11" ).on('change',
      function(dt)
      {
        $("span#sliderval").text($(dt.target).val())
      })

			$("input[name='interval']").change(function(){
				$(this).parent().siblings().children().attr("checked", false);
				$(this).attr("checked", true);
				showStats()
			
			})


}
)

function preSubmit(dt)
{

    var loglist = []
    var startstr = $("#startdate").val()
    var endstr = $("#enddate").val()
		var start = new Date(startstr);
		var end = new Date(endstr);

    for(var ii=0; ii<6; ii++)
    {
        var value = $("select#log"+ii).val()
        if(value != null)
            loglist[ii-1] = value;
    }
    if(!startstr)
    {
        triggerModalError("Submit Error", "Please enter a start date.");
        return;
    }

    if(!endstr)
    {
        triggerModalError("Submit Error", "Please enter an end date.");
        return;
    }

    if(loglist.length == 0)
    {
      triggerModalError("Submit Error", "Please select at least one log.");
			return;
    }


		if (start > end)
		{
      triggerModalError("Submit Error", "The start date needs to be after end date.");
			return;
		}
		
		nsamples = (end-start)/sampleInterval()
		if(nsamples < 1)
		{
			triggerModalError("Submit Error", "Your sample interval is too large! You will recieve no data!");
		}

		var days = (end-start)/(24*3600*1000);
		console.log(days);
		if(days > 30*9)
		{
			triggerModalError("Submit Error", "Querying more than 270 days of data at once is not allowed! You will have to submit multiple jobs.");
		}
		else if( days > 30*3)
		{
			$("button#btn-submit").click(function() 
				{
					submit(start, end, nsamples, loglist);
				});
			triggerModalPresubmit("Submit", "You are querying "+ days +" days. This may take a while." );

		}
		else
			submit(start, end, nsamples, loglist);

}

function submit(start, end, nsamples, loglist)
{	

	$('#submit').attr("disabled", "disabled")
	
    output = {
        ds_names: loglist,
        nsamples: nsamples,
        startdate: start,
        enddate: end
    }

    $.ajax
    (
    {
        url: "recent/json",
        data:JSON.stringify(output),
        content: "application/json",
        type: "post",
        success:function(dt)
        {
            console.log("submision success")
            console.log(dt)
            window.location.href = "/job/data.html?jobid="+dt['info']['jobid']
        },
    }

    )

}

function triggerModalError(title, msg)
{
    $('#error_modal_title').text(title);
    $('#error_modal_msg').text(msg);
    $('div#error').modal();
}


function triggerModalPresubmit(title, msg)
{
    $('#presubmit_modal_title').text(title);
    $('#presubmit_modal_msg').text(msg);
    $('div#beforesubmit').modal();
}

function showStats()
{

		mult = parseFloat($('#sample_interval').val())
		start = new Date($('#startdate').val());
		end = new Date($('#enddate').val());
		nsamples = (end-start)/sampleInterval();
		$("p#stats").text("This will give approximately "+Math.round(nsamples)+" samples")

}

function sampleInterval()
{
		map = {
			secs	:1000,
			hours	:3600*1000,
			days	:24*3600*1000
		};

		mult = parseFloat($('#sample_interval').val())	
		$("input[name='interval']").each(function(){
			if($(this).attr("checked"))
				interval=map[this.id]
		});

		return interval*mult;

}


