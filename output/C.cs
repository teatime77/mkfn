using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;

namespace MkFn {
    public partial class MakeCode {
        public MkFn theMkFn;
        public const string NL = "\r\n";
        Dictionary<LINQ, string> LinqValue = new Dictionary<LINQ, string>();

        public MakeCode(MkFn mkfn) {
            theMkFn = mkfn;
        }

        public string MakeC() {
            StringWriter sw = new StringWriter();

            foreach (Class cls in theMkFn.Layers) {

            }


            return sw.ToString();
        }

        public string Nest(int nest) {
            return new string(' ', 4 * nest);
        }

        public string TypeCode(Class cls) {
            return "TYPE";
        }

        public void ForHeadCode(Variable loop_var, StringWriter sw, int nest) {
            if (!Term.IsRange(loop_var.Domain)) {
                throw new Exception();
            }

            Apply rng = loop_var.Domain as Apply;

            if (rng.Args.Length == 1) {

                sw.WriteLine("{0}for(int {1} = 0; {1} < {2}; {1}++){{", Nest(nest), loop_var.Name, TermCode(rng.Args[0]));
            }
            else if (rng.Args.Length == 2) {

                sw.WriteLine("{0}for(int {1} = {2}; {1} < {3}; {1}++){{", Nest(nest), loop_var.Name, TermCode(rng.Args[0]), TermCode(rng.Args[1]));
            }
            else {

                throw new Exception();
            }
        }

        public string StatementCode(Statement stmt, int nest) {
            StringWriter sw = new StringWriter();

            if (stmt is Assignment || stmt is Return) {

                // すべての代入文のリスト
                List<LINQ> lnks = new List<LINQ>();
                MkFn.Navi(stmt,
                    delegate (object obj) {
                        if (obj is LINQ) {
                            lnks.Add(obj as LINQ);
                        }
                    });

                int tmp_cnt = 0;


                foreach (LINQ lnq in lnks) {

                    // 集計用の変数
                    tmp_cnt++;
                    string tmp_name = string.Format("" ,tmp_cnt);
                    LinqValue.Add(lnq, tmp_name);

                    if(lnq.Aggregate.VarRef == theMkFn.SumFnc) {

                        sw.WriteLine("{0}{1} {2} = 0;", Nest(nest), TypeCode(lnq.TypeTerm), tmp_name);
                    }
                    else if (lnq.Aggregate.VarRef == theMkFn.ProdFnc) {

                        sw.WriteLine("{0}{1} {2} = 1;", Nest(nest), TypeCode(lnq.TypeTerm), tmp_name);
                    }
                    else if (lnq.Aggregate.VarRef == theMkFn.MaxFnc) {

                        sw.WriteLine("{0}{1} {2} = ;", Nest(nest), TypeCode(lnq.TypeTerm), tmp_name);
                    }

                    foreach (Variable loop_var in lnq.Variables) {
                        ForHeadCode(loop_var, sw, nest);
                    }

                    if (lnq.Aggregate.VarRef == theMkFn.SumFnc) {

                        sw.WriteLine("{0}{1} += {2};", Nest(nest+1), tmp_name, TermCode(lnq.Select));
                    }
                    else if (lnq.Aggregate.VarRef == theMkFn.ProdFnc) {

                        sw.WriteLine("{0}{1} *= {2};", Nest(nest+1), tmp_name, TermCode(lnq.Select));
                    }
                    else if (lnq.Aggregate.VarRef == theMkFn.MaxFnc) {

                        sw.WriteLine("{0}{1} {2} = ;", Nest(nest+1), tmp_name, TermCode(lnq.Select));
                    }

                    foreach (Variable loop_var in lnq.Variables) {
                        sw.WriteLine("}");
                    }
                }

                if (stmt is Assignment) {
                    Assignment asn = stmt as Assignment;

                    sw.WriteLine( Nest(nest) + TermCode(asn.Left) + " = " + TermCode(asn.Right) + ";");
                }
                else {
                    sw.WriteLine( Nest(nest) + "return " + TermCode((stmt as Return).Value) + ";");
                }
            }
            else if(stmt is ForEach) {
                ForEach fe = stmt as ForEach;

                ForHeadCode(fe.LoopVariable, sw, nest);

                foreach (Statement stmt2 in (stmt as BlockStatement).Statements) {
                    sw.Write(StatementCode(stmt2, nest + 1));
                }

                sw.WriteLine(Nest(nest) + "}");
            }
            else if (stmt is BlockStatement) {

                foreach(Statement stmt2 in (stmt as BlockStatement).Statements) {
                    sw.Write(StatementCode(stmt2, nest + 1));
                }
            }
            else {
                throw new Exception();
            }

            return sw.ToString();
        }

        public string TermCode(Term t1) {
            if (!(t1 is Number)) {

                if (t1.Value == 1) {

                    return TermCodeBody(t1);
                }
                else if (t1.Value == -1) {

                    return "- " + TermCodeBody(t1);
                }
                else {

                    return t1.Value.ToString() + " * " + TermCodeBody(t1);
                }
            }
            else {

                return TermCodeBody(t1);
            }
        }

        string TermCodeBody(Term t1) {
            if (t1 is LINQ) {


                return LinqValue[t1 as LINQ];
            }
            else {
                return t1.ToString();
            }
        }

        public string VariableCode(Variable v) {
            return TypeCode(v.TypeVar) + " " + v.Name;
        }

        public string FunctionCode(Function fnc) {
            StringWriter sw = new StringWriter();

            sw.Write("{0} {1}", TypeCode(fnc.TypeVar), fnc.Name);
            sw.WriteLine("(" + string.Join(", ", from v in fnc.Params select VariableCode(v)) + "){");
            sw.Write(StatementCode(fnc.BodyStatement, 1));
            sw.WriteLine("}");

            return sw.ToString();
        }

        public string FieldCode(Variable fld) {
            return TypeCode(fld.TypeVar) + " " + fld.Name + ";\r\n";
        }

        public void ClassCode(Class cls, out string header, out string body) {
            header = "struct " + cls.Name + " {\r\n" + string.Join("", from fld in cls.Fields select FieldCode(fld)) + "}\r\n";
            body = string.Join("\r\n", from fnc in cls.Functions select FunctionCode(fnc));
        }
    }
}
