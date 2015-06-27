namespace InferNetHOL.CSharp
{
    using System;
    using System.Threading;
    using System.Threading.Tasks;

    public interface IExample
    {
        string Title { get; }

        void Run();

        LabelledSeries<Tuple<double, double>>[] Series { get; }
    }
}
