using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Diagnostics;

namespace MkFn {
    public partial class MkFn {
        /*
         * 名前解決
         */
        void ResolveName() {
            // アプリのクラスに対し
            foreach (Class cls in Layers) {
                Debug.WriteLine("layer : {0}", cls.Name, "");

                Traverse(cls,
                    null,
                    delegate (object obj) {
                        if (obj is Reference) {
                            // 変数参照の場合

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

        public List<Variable> GetVariables(Term trm) {
            List<Variable> vars = new List<Variable>();

            for (object obj = trm.Parent; ; ) {
                Debug.Assert(obj != null);

                object parent = null;
                if (obj is Term) {
                    // 項の場合

                    if (obj is LINQ) {
                        // LINQの場合

                        vars.AddRange((obj as LINQ).Variables);
                    }

                    parent = (obj as Term).Parent;
                }
                else if (obj is Statement) {
                    // 文の場合

                    if (obj is ForEach) {
                        // foreachの場合

                        vars.AddRange((obj as ForEach).LoopVariables);
                    }

                    parent = (obj as Statement).ParentStmt;
                }
                else if (obj is Variable) {
                    // 変数の場合

                    if (obj is Function) {
                        // 関数の場合

                        vars.AddRange((obj as Function).Params);
                    }

                    parent = (obj as Variable).ParentVar;
                }
                else if (obj is Class) {
                    // クラスの場合

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

        /*
        型推論
        */
        void TypeInference(object root) {
            Traverse(root,
                null,
                delegate (object obj) {
                    if (obj is Term) {
                        // 項の場合

                        Term trm = obj as Term;
                        if (trm is Reference) {
                            // 変数参照の場合

                            Reference rf = trm as Reference;

                            if (rf.VarRef == NewFnc) {
                                return;
                            }

                            if (rf.Indexes == null) {

                                rf.TypeTerm = rf.VarRef.TypeVar;
                            }
                            else {

                                rf.TypeTerm = (rf.VarRef.TypeVar as ArrayType).ElementType;
                            }
                        }
                        else if (trm is Number) {
                            // 数値定数の場合

                        }
                        else if (trm is Apply) {
                            // 関数適用の場合

                            Apply app = trm as Apply;

                            if (IsNew(app)) {


                                trm.TypeTerm = GetArrayType(app.NewClass, app.Args.Length);
                            }
                            else if (app.FunctionApp.VarRef.TypeVar == ArgClass) {

                                Class tp1 = app.Args[0].TypeTerm;
                                for (int i = 1; i < app.Args.Length; i++) {
                                    Class tp2 = app.Args[i].TypeTerm;
                                    if (MkFn.Singleton.NumberTypeOrder(tp1) < MkFn.Singleton.NumberTypeOrder(tp2)) {

                                        tp1 = tp2;
                                    }
                                }

                                trm.TypeTerm = tp1;
                            }
                            else {
                                trm.TypeTerm = app.FunctionApp.VarRef.TypeVar;
                            }
                        }
                        else if (trm is LINQ) {
                            // LINQの場合

                            LINQ lnq = trm as LINQ;

                            Debug.Assert(lnq.Aggregate != null);
                            trm.TypeTerm = lnq.Select.TypeTerm;
                        }
                        else {
                            Debug.Assert(false);
                        }
                        Debug.Assert(trm.TypeTerm != null);
                    }
                    else if (obj is Variable) {
                        // 変数の場合

                        Variable va = obj as Variable;

                        if (va.TypeVar == null) {
                            if (va.Domain == null) {
                                throw new Exception();
                            }
                            else {

                                if (va.Domain.TypeTerm.DimCnt != 1) {
                                    throw new Exception();
                                }

                                va.TypeVar = (va.Domain.TypeTerm as ArrayType).ElementType;
                                Debug.Assert(va.TypeVar != null);
                            }
                        }
                    }
                });
        }
    }
}
