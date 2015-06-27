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
    public class CurveFitting : ExampleBase
    {
        public CurveFitting() : base("Curve fitting") { }

        public override void Run()
        {
            var engine = new InferenceEngine();
            var series = new List<LabelledSeries<Tuple<double, double>>>();

            var rand = new Random();
            var input = Enumerable.Range(-100, 200).Select(x => Convert.ToDouble(x) / 100.0).ToArray();
            var expected = input.Select(x => Math.Sin(2 * Math.PI * x)).ToArray();
            var observed = expected.Select(x => x + (rand.NextDouble() - 0.5)).ToArray();

            series.Add(new LabelledSeries<Tuple<double, double>>("Actual sin(2x*Pi)", input.Zip(expected, (f, s) => Tuple.Create(f, s))));

            var m = Variable.New<int>();
            var rR = new Range(input.Length).Named("r");
            var rM = new Range(m).Named("M");
            var X = Variable.Array<double>(rR, rM).Named("X");
            var w = Variable.Array<double>(rM).Named("W");
            var noise = Variable.GammaFromShapeAndScale(1,5);
            w[rM] = Variable.GaussianFromMeanAndPrecision(0, 0.01).ForEach(rM);
            var y = Variable.Array<double>(rR).Named("Y");
            using (Variable.ForEach(rR))
            {
                var prods = Variable.Array<double>(rM);
                using (Variable.ForEach(rM))
                {
                    prods[rM] = X[rR, rM] * w[rM];
                }

                y[rR] = Variable.Sum(prods);
            }

            for (var i = 3; i < 8; i++)
            {
                m.ObservedValue = i;
                var inputArr = new double[input.Length, i];
                for (var r = 0; r < input.Length; r++)
                {
                    for (var c = 0; c < i; c++)
                    {
                        inputArr[r, c] = Math.Pow(input[r], c);
                    }
                }

                X.ObservedValue = inputArr;
                y.ObservedValue = observed;

                var posteriorW = engine.Infer<DistributionStructArray<Gaussian, double>>(w);
                var weights = posteriorW.Select(d => d.GetMean()).ToArray();
                var implied = new double[input.Length];
                for (var r = 0; r < implied.Length; r++)
                {
                    var val = 0.0;
                    for (var c = 0; c < i; c++)
                    {
                        val += Math.Pow(input[r], c) * weights[c];
                    }

                    implied[r] = val;
                }

                var polinomialLabelParts = new List<string>();
                for (var c = 0; c < i; c++)
                {
                    polinomialLabelParts.Add(Math.Round(weights[c], 4) + "*x^" + c);
                }

                var polinomialLabel = string.Join("+", polinomialLabelParts);

                series.Add(new LabelledSeries<Tuple<double, double>>(string.Format("Inferred {0}", polinomialLabel), input.Zip(implied, (f, s) => Tuple.Create(f, s))) { IsScatter = false });
            }
            
            series.Add(new LabelledSeries<Tuple<double, double>>("Data", input.Zip(observed, (f, s) => Tuple.Create(f, s))) { IsScatter = true });

            this.Series = series.Select(s => new LabelledSeries<Tuple<double, double>>(s.Label, s.Series.Skip(50).Take(100)) { IsScatter = s.IsScatter }).ToArray();
        }
    }
}
