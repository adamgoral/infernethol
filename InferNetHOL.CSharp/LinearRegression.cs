using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace InferNetHOL.CSharp
{
    public class LinearRegression : ExampleBase
    {
        public LinearRegression() : base("Linear regression") { }

        public override void Run()
        {
            var rangeMin = -10;
            var interval = 0.1;
            var observationSize = 100;
            var aActual = 0.2;
            var bActual = 2.3;
            var rand = new System.Random();
            var actuals = Enumerable.Range(rangeMin, observationSize)
                    .Select(i => i * interval)
                    .Select(i => Tuple.Create((double) i, bActual * i + aActual))
                    .ToArray();
            var samples = actuals.Select(tuple => Tuple.Create(tuple.Item1, tuple.Item2 + ((rand.NextDouble() - 0.5) * 10))).ToArray();

            var series = new List<LabelledSeries<Tuple<double, double>>>();
            series.Add(new LabelledSeries<Tuple<double, double>>(string.Format("Actual a+bx a={0} b={1}", aActual, bActual), actuals));

            var aPrior = Variable.GaussianFromMeanAndPrecision(0, 0.01).Named("aPrior");
            var bPrior = Variable.GaussianFromMeanAndPrecision(0, 0.01).Named("bPrior");
            var noisePrior = Variable.GammaFromShapeAndScale(1, 5).Named("noisePrior");
            var obsRange = new Range(samples.Length);
            var xArray = Variable.Array<double>(obsRange);
            var exprArray = Variable.Array<double>(obsRange);
            using (Variable.ForEach(obsRange))
            {
                exprArray[obsRange] = Variable.GaussianFromMeanAndPrecision(aPrior + xArray[obsRange] * bPrior, noisePrior);
            }

            xArray.ObservedValue = samples.Select(t => (double)t.Item1).ToArray();
            exprArray.ObservedValue = samples.Select(t => t.Item2).ToArray();

            var engine = new InferenceEngine();
            var aPosterior = engine.Infer<Gaussian>(aPrior);
            var bPosterior = engine.Infer<Gaussian>(bPrior);
            var noisePosterior = engine.Infer<Gamma>(noisePrior);

            var aInferred = aPosterior.GetMean();
            var bInferred = bPosterior.GetMean();
            var inferred = Enumerable.Range(rangeMin, observationSize)
                            .Select(i => i * interval)
                            .Select(i => Tuple.Create((double)i, bInferred * i + aInferred))
                            .ToArray();

            series.Add(new LabelledSeries<Tuple<double, double>>(string.Format("Inferred a+bx a={0} b={1}", Math.Round(aInferred, 4), Math.Round(bInferred, 4)), inferred));

            series.Add(new LabelledSeries<Tuple<double, double>>(string.Format("Data", aActual, bActual), samples) { IsScatter = true }); 

            this.Series = series.ToArray();
        }
    }
}
