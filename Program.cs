using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mkfn {
    partial class Program {
        
        static void Main(string[] args) {
            mkfn.Singleton = new mkfn();
            mkfn.Singleton.Main();
        }
    }
}
