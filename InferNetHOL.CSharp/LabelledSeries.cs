using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace InferNetHOL.CSharp
{
    public class LabelledSeries<T>
    {
        public LabelledSeries(string label, IEnumerable<T> series)
        {
            this.Label = label;
            this.Series = series.ToArray();
        }

        public string Label { get; private set; }

        public T[] Series { get; private set; }

        public bool IsScatter { get; set; }
    }
}
