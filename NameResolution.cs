using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Diagnostics;

namespace mkfn {
    public partial class mkfn {
        /*
         * 名前解決
         */
        void ResolveName() {
            // アプリのクラスに対し
            foreach (Class cls in AppClasses) {
                Debug.WriteLine("layer : {0}", cls.Name, "");

                var list = from f in cls.Functions where f.Name == "Forward" && f.Statement != null select f;
                if (list.Any()) {

                    Function fnc = list.First();

                    Navi(fnc.Statement,
                        null,
                        delegate (object obj) {
                            if (obj is Reference) {

                                Reference ref1 = obj as Reference;

                                if (ref1.VarRef == null) {

                                    var vars = from v in GetVariables(ref1) where v.Name == ref1.Name select v;
                                    if (vars.Any()) {
                                        ref1.VarRef = vars.First();
                                    }
                                    else {
                                        Debug.WriteLine("未定義 {0}", ref1.Name, "");
                                    }
                                }
                            }
                        });
                }
            }
        }



        public List<Variable> GetVariables(Term trm) {
            List<Variable> vars = new List<Variable>();

            for (object obj = trm.Parent; ; ) {
                Debug.Assert(obj != null);

                object parent = null;
                if (obj is Term) {

                    if (obj is LINQ) {
                        vars.AddRange((obj as LINQ).Variables);
                    }

                    parent = (obj as Term).Parent;
                }
                else if (obj is Statement) {

                    if (obj is ForEach) {

                        vars.Add((obj as ForEach).LoopVariable);
                    }

                    parent = (obj as Statement).ParentStmt;
                }
                else if (obj is Variable) {

                    if (obj is Function) {

                        vars.AddRange((obj as Function).Params);
                    }

                    parent = (obj as Variable).ParentVar;
                }
                else if (obj is Class) {

                    Class cls = (obj as Class);

                    vars.AddRange(cls.Fields);
                    vars.AddRange(cls.Functions);

                    if(cls.Parent == null) {

                        break;
                    }
                    parent = cls.Parent;
                }
                else {

                    Debug.Assert(false);
                }

                Debug.Assert(parent != null);
                obj = parent;
            }

            return vars;
        }
    }
}
