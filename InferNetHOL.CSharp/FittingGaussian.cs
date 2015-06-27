namespace InferNetHOL.CSharp
{
    using MicrosoftResearch.Infer;
    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Models;
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Threading;
    using System.Threading.Tasks;
    
    public class FittingGaussian : ExampleBase
    {
        public FittingGaussian() : base("Fitting Gaussian") { }

        public override void Run()
        {
            var sourceMean = 11.4;
            var sourcePrecision = 0.01;
            var source = Gaussian.FromMeanAndPrecision(sourceMean, sourcePrecision);
            var series = new List<LabelledSeries<Tuple<double, double>>>();
            series.Add(new LabelledSeries<Tuple<double, double>>(string.Format("Actual mean {0} precision {1}", source.GetMean(), source.Precision), Enumerable.Range(-30, 80).Select(x => Tuple.Create((double)x, Math.Exp(source.GetLogProb(x))))));

            // Prior distributions
            var meanPriorDistr = Gaussian.FromMeanAndPrecision(0, 0.01);
            var precisionPriorDistr = Gamma.FromMeanAndVariance(2, 5);

            var meanPrior = Variable.Random(meanPriorDistr).Named("mean");
            var precPrior = Variable.Random(precisionPriorDistr).Named("precision");
            var tv = Variable.New<int>();
            var tr = new Range(tv).Named("tr");
            var engine = new InferenceEngine();
            var xv = Variable.GaussianFromMeanAndPrecision(meanPrior, precPrior).Named("xv");
            var xs = Variable.Array<double>(tr).Named("xs");
            xs[tr] = xv.ForEach(tr);

            var maxSampleSize = 250;
            var sampleData = Enumerable.Range(0, maxSampleSize + 1).Select(_ => source.Sample()).ToArray();

            for (var i = 50; i <= maxSampleSize; i += 50)
            {
                tv.ObservedValue = i;
                xs.ObservedValue = sampleData.Take(i).ToArray();
                var meanPost = engine.Infer<Gaussian>(meanPrior);
                var precPost = engine.Infer<Gamma>(precPrior);
                var estimateDist = Gaussian.FromMeanAndPrecision(meanPost.GetMean(), precPost.GetMean());
                series.Add(new LabelledSeries<Tuple<double, double>>(string.Format("Implied mean {0} precision {1} with {2} samples", Math.Round(estimateDist.GetMean(), 4), Math.Round(estimateDist.Precision, 4), i), Enumerable.Range(-30, 80).Select(x => Tuple.Create((double)x, Math.Exp(estimateDist.GetLogProb(x))))));
            }
            
            this.Series = series.ToArray();
        }
    }
}
