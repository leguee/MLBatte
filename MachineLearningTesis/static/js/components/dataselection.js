define([], function() {
    var DataSelectionPanel = React.createClass({
        render: function(){
            return (
                <select className="input-medium" id="selectDataSource"/>
            );
        },

        componentDidMount: function() {
            var handle = this.props.handleSelection;

            $.get( "/csvdata", function( data ) {
                var result = $.parseJSON(data);
                var dataSourceSelect = $("#selectDataSource");
                dataSourceSelect.select2({
                    data: result,
                    tags: "true",
                    width: "256px",
                    placeholder: "Seleccione una fuente de datos",
                    allowClear: true
                });

                dataSourceSelect.val(null).trigger("change");

                dataSourceSelect.change(function(){
                    console.log("New Data Source selected : " + $(this).val());
                    handle($(this).val());
                });
            });  
        }
    });

    return DataSelectionPanel;
});