define(["comp/ml/mlview",
        "ml/viz/clsviz"
        ], function(ML, Viz) {

    var name = "NeuralNetwork";
    var shortName = "neuralnetwork";

    var buildTrainParameterHandler = function(trainPara, dataBinding){
        trainPara.label = dataBinding.Label;
        trainPara.features = dataBinding.Features.join();
    };

    var renderPredictVizHandler = function(option, dataBinding) {
        option.target = dataBinding.Target;
        var viz = new Viz(option);
        viz.render();
    };

    var NeuralNetworkPage = React.createClass({
        render: function() {
            return ( 
                <ML modelName={name} modelShortName={shortName} buildTrainParameterHandler={buildTrainParameterHandler} renderPredictVizHandler={renderPredictVizHandler} />
            );
        },
        componentDidMount: function() {
        }
    });

    return NeuralNetworkPage;
});