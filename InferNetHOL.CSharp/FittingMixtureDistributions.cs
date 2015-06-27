using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace InferNetHOL.CSharp
{
    public class FittingMixtureDistributions : ExampleBase
    {
        public FittingMixtureDistributions() : base("Fitting mixture distribution") { }

        public override void Run()
        {
            var numOfDistRange = new Range(2).Named("numOfDistRange");
            var mixingCoefficients = Variable.Array<double>(numOfDistRange).Named("mixingCoefficients");
            var meanPriors = Variable.Array<double>(numOfDistRange);
            var precisionPriors = Variable.Array<double>(numOfDistRange);
            var distributions = Variable.Array<double>(numOfDistRange);
            using (Variable.ForEach(numOfDistRange))
            {
                meanPriors[numOfDistRange] = Variable.GaussianFromMeanAndPrecision(0, 0.01);
                precisionPriors[numOfDistRange] = Variable.GammaFromShapeAndScale(1, 5);
                distributions[numOfDistRange] = Variable.GaussianFromMeanAndPrecision(meanPriors[numOfDistRange], precisionPriors[numOfDistRange]);
            }

            throw new NotImplementedException();
        }
    }
}
