namespace InferNetHOL.CSharp
{
    using MicrosoftResearch.Infer;
    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Distributions.Kernels;
    using MicrosoftResearch.Infer.Maths;
    using MicrosoftResearch.Infer.Models;
    using SmartTrader.Data;
    using SmartTrader.Domain;
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Threading;
    using System.Threading.Tasks;

    public class GaussianProcessExample : ExampleBase
    {
        public GaussianProcessExample()
            : base("Gaussian process")
        {
        }

        public override void Run()
        {
            var prices = GetPrices("SPY").Select(t => t.Item2).ToArray();
            var inputs = prices.Take(prices.Length - 1).ToArray();
            var inputVectors = inputs.Select(i => Vector.FromArray(new[] { i })).ToArray();
            var outputs = prices.Skip(1).ToArray();

            // Set up the GP prior, which will be filled in later
            var prior = Variable.New<SparseGP>().Named("prior");
            
            // The sparse GP variable - a distribution over functions
            var f = Variable<IFunction>.Random(prior).Named("f");

            // The locations to evaluate the function
            var x = Variable.Observed(inputVectors).Named("x");
            var j = x.Range.Named("j");
            var y = Variable.Observed(outputs, j).Named("y");
            y[j] = Variable.GaussianFromMeanAndVariance(Variable.FunctionEvaluate(f, x[j]), 0.1);
            var kf = new SquaredExponential(-0.5);

            // The basis
            var rand = new Random(Environment.TickCount);
            var basis = Enumerable.Range(1, 10).Select(i => Vector.FromArray(new double[1] {i*10})).ToArray();
            //var basis = new Vector[] {
            //    Vector.FromArray(new double[1] {80}),
            //    Vector.FromArray(new double[1] {90}),
            //    Vector.FromArray(new double[1] {100})
            //};

            var gp = new GaussianProcess(new ConstantFunction(0), kf);

            prior.ObservedValue = new SparseGP(new SparseGPFixed(gp, basis));
            var engine = new InferenceEngine(new ExpectationPropagation());
            var sgp = engine.Infer<SparseGP>(f);

            var means = sgp.Mean(inputVectors).ToArray();
            var stdDevs = inputVectors.Select(iv => Math.Sqrt(sgp.Variance(iv))).ToArray();

            this.Series = new[]
            { 
                new LabelledSeries<Tuple<double,double>>(
                    "input",
                    Enumerable.Range(0,inputs.Length)
                    .Select(i=> Tuple.Create((double)i, inputs[i]))),
                new LabelledSeries<Tuple<double,double>>(
                    "infered mean",
                    Enumerable.Range(0,inputs.Length)
                    .Select(i=> Tuple.Create((double)i, means[i]))),
                new LabelledSeries<Tuple<double,double>>(
                    "infered stddev",
                    Enumerable.Range(0,inputs.Length)
                    .Select(i=> Tuple.Create((double)i, stdDevs[i]))),
            };
        }

        private static string[] GetInstruments()
        {
            var instrumentsSource = new LocalInstrumentSource(new Uri(Properties.Settings.Default.instrumentsSourcePath));
            return instrumentsSource
                .GetInstruments(CancellationToken.None)
                .Result
                .Select(i => i.Symbol)
                .ToArray();
        }

        private static DateTime[] GetDates(IEnumerable<IEnumerable<DateTime>> dateGroups)
        {
            var hashSets = dateGroups.Select(ds => new HashSet<DateTime>(ds)).ToArray();
            var union = new HashSet<DateTime>(hashSets.SelectMany(set => set));
            var result = new List<DateTime>();
            foreach (var item in union)
            {
                if (hashSets.All(hs => hs.Contains(item)))
                {
                    result.Add(item);
                }
            }

            return result.ToArray();
        }

        private static Tuple<DateTime, double>[] GetPrices(string instrument)
        {
            var pricesSource = new LocalMarketDataStore(new Uri(Properties.Settings.Default.priceSourcePath));
            var data = pricesSource
                .GetAsync(
                    new Instrument(instrument),
                    DateTime.MinValue,
                    DateTime.MaxValue,
                    CancellationToken.None)
                .Result
                .ToArray();

            var result = data
                .Select(d => Tuple.Create(d.Date, d.AdjustedClose))
                .ToArray();

            return result;
        }
    }
}
