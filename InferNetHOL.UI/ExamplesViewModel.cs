using InferNetHOL.CSharp;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;

namespace InferNetHOL.UI
{
    public class ExamplesViewModel : INotifyPropertyChanged
    {
        private IExample selectedExample;

        public event PropertyChangedEventHandler PropertyChanged;

        public List<IExample> Examples { get; private set; }

        public IExample SelectedExample
        {
            get { return this.selectedExample; }
            set
            {
                if (this.selectedExample != value)
                {
                    this.selectedExample = value;
                    this.OnPropertyChanged("SelectedExample");
                    this.RunExample.OnCanExecuteChanged();
                }
            }
        }

        public DelegateCommand<IExample> RunExample { get; private set; }

        public ExamplesViewModel()
        {
            this.Examples = new List<IExample>(LoadExamples());
            this.RunExample = new DelegateCommand<IExample>(e => e.Run(), e => e != null);
        }

        private void OnPropertyChanged(string propertyName)
        {
            var handler = this.PropertyChanged;
            if (handler != null)
            {
                handler(this, new PropertyChangedEventArgs(propertyName));
            }
        }

        private static IEnumerable<IExample> LoadExamples()
        {
            return AppDomain.CurrentDomain.GetAssemblies().SelectMany(LoadExamples);
        }

        private static IEnumerable<IExample> LoadExamples(Assembly assembly)
        {   var exampleType = typeof(IExample);
            return assembly.GetTypes()
                .Where(t => !t.IsInterface && !t.IsAbstract)
                .Where(t => exampleType.IsAssignableFrom(t))
                .Select(t => (IExample)t.Assembly.CreateInstance(t.FullName));
        }
    }
}
