$(function()
{
    $.ajax(
    {
        url:'/info/system2log.json',
        success:
        function(data, status, jqXHR)
        {
            if("error" in data)
            {
                trigger_modal("Error", data["error"])
            }
            for(ii=1; ii<6; ii++)
            {
                $("select#system"+ii).on('change', function(dt)
                    {
                        var logs = data[$(dt.target).val()]

                        which = dt.target.id[dt.target.id.length-1]
                        $("select#log"+which+" > option").remove()
                        $("select#log"+which).append('<option></option>')
                        for(log in logs)
                        {

                            $("select#log"+which).append('<option>'+logs[log]+'</option>')
                        }
                    }
                    )
                $("select#system"+ii+"> option").remove()
                $("select#system"+ii).append('<option>'+'Choose...'+'</option>')
                for(system in data)
                {
                        $("select#system"+ii)
                            .append('<option>'+system+'</option>')

                }
            }
         },
         failure: function(fail)
         {
            trigger_modal("Failed to contact server", fail)
         }
         })
})



