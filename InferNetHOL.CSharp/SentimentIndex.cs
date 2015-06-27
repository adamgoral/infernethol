namespace InferNetHOL.CSharp
{
    using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

    public class SentimentIndex : ExampleBase
    {
        private readonly Variable<int> numberOfTrainingItems, numberOfTestingItems;
        private readonly VariableArray<Vector> trainingInputs, testingInputs;
        private readonly VariableArray<bool> trainingOutputs, testingOutputs;
        private readonly Variable<Vector> weights;
        private readonly Variable<VectorGaussian> weightsPosteriorDistribution;
        private readonly int numberOfFeatures = 10;
        private readonly double noise = 1;
        private readonly InferenceEngine trainingEngine, testingEngine;

        public SentimentIndex() : base("Sentiment Index")
        {
            numberOfTrainingItems = Variable.New<int>();
            var rangeOfTrainingItems = new Range(numberOfTrainingItems);
            trainingInputs = Variable.Array<Vector>(rangeOfTrainingItems);
            trainingOutputs = Variable.Array<bool>(rangeOfTrainingItems);

            weights = Variable.Random(new VectorGaussian(Vector.Zero(numberOfFeatures), PositiveDefiniteMatrix.Identity(numberOfFeatures)));

            using (Variable.ForEach(rangeOfTrainingItems))
            {
                trainingOutputs[rangeOfTrainingItems] = Variable.IsPositive(Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(weights, trainingInputs[rangeOfTrainingItems]), noise));
            }

            trainingEngine = new InferenceEngine();
            trainingEngine.ShowProgress = false;

            numberOfTestingItems = Variable.New<int>();
            var rangeOfTestingItems = new Range(numberOfTestingItems);
            testingInputs = Variable.Array<Vector>(rangeOfTestingItems);
            testingOutputs = Variable.Array<bool>(rangeOfTestingItems);

            weightsPosteriorDistribution = Variable.New<VectorGaussian>();
            var testWeights = Variable<Vector>.Random(weightsPosteriorDistribution);

            using (Variable.ForEach(rangeOfTestingItems))
            {
                testingOutputs[rangeOfTestingItems] = Variable.IsPositive(Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(testWeights, testingInputs[rangeOfTestingItems]), noise));
            }

            testingEngine = new InferenceEngine();
            testingEngine.ShowProgress = false;
        }

        public override void Run()
        {
            throw new NotImplementedException();
        }

        private void Train(Vector[] inputs, bool[] outputs)
        {
            numberOfTrainingItems.ObservedValue = inputs.Length;
            trainingInputs.ObservedValue = inputs;
            trainingOutputs.ObservedValue = outputs;
            weightsPosteriorDistribution.ObservedValue = trainingEngine.Infer<VectorGaussian>(testingOutputs);
        }
    }
}
