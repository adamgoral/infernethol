using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace InferNetHOL.CSharp
{
    public abstract class ExampleBase : IExample, INotifyPropertyChanged
    {
        public ExampleBase(string title)
        {
            this.Title = title;
        }

        public string Title { get; private set; }

        public abstract void Run();

        private LabelledSeries<Tuple<double, double>>[] series;

        public LabelledSeries<Tuple<double, double>>[] Series 
        {
            get { return this.series; }
            protected set
            {
                this.series = value;
                this.OnPropertyChanged("Series");
            }
        }

        public event PropertyChangedEventHandler PropertyChanged;

        private void OnPropertyChanged(string propertyName)
        {
            var handler = this.PropertyChanged;
            if (handler != null)
            {
                handler(this, new PropertyChangedEventArgs(propertyName));
            }
        }
    }
}
