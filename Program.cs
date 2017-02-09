using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MkFn {
    partial class Program {
        
        static void Main(string[] args) {
            MkFn.Singleton = new MkFn();
            MkFn.Singleton.Main();
        }
    }
}
