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
            Debug.Assert(cls != null);

            if(cls.DimCnt == 0) {
                return cls.Name;
            }
            else {

                return cls.Name + "[" +  string.Join(",", from x in MkFn.Range(cls.DimCnt - 1) select "") + "]";
            }
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
                // 代入文かreturnの場合

                sw.WriteLine("");
                sw.WriteLine(Nest(nest) + "// " + stmt.ToString());

                // すべての代入文のリスト
                List<LINQ> lnks = new List<LINQ>();
                MkFn.Traverse(stmt,
                    delegate (object obj) {
                        if (obj is LINQ) {
                            // LINQの場合

                            lnks.Add(obj as LINQ);
                        }
                    });

                int tmp_cnt = 0;


                foreach (LINQ lnq in lnks) {

                    // 集計用の変数
                    tmp_cnt++;
                    string tmp_name = string.Format("_wk{0}" ,tmp_cnt);
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
                        sw.WriteLine(Nest(nest) + "}");
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
                // foreachの場合

                ForEach fe = stmt as ForEach;

                ForHeadCode(fe.LoopVariable, sw, nest);

                foreach (Statement stmt2 in (stmt as BlockStatement).Statements) {
                    sw.Write(StatementCode(stmt2, nest + 1));
                }

                sw.WriteLine(Nest(nest) + "}");
            }
            else if (stmt is BlockStatement) {
                // ブロック文の場合

                foreach (Statement stmt2 in (stmt as BlockStatement).Statements) {
                    sw.Write(StatementCode(stmt2, nest + 1));
                }
            }
            else {
                throw new Exception();
            }

            return sw.ToString();
        }

        public string TermCode(Term trm) {
            if (!(trm is Number)) {
                // 数値定数でない場合

                if (trm.Value == 1) {

                    return TermCodeBody(trm);
                }
                else if (trm.Value == -1) {

                    return "- " + TermCodeBody(trm);
                }
                else {

                    return trm.Value.ToString() + " * " + TermCodeBody(trm);
                }
            }
            else {

                return TermCodeBody(trm);
            }
        }

        string TermCodeBody(Term trm) {
            if (trm is Reference) {
                // 変数参照の場合

                Reference rf = trm as Reference;

                if (rf.Indexes == null) {
                    return rf.Name;
                }
                else {
                    return rf.Name + string.Join("", from x in rf.Indexes select "[" + TermCode(x) + "]");
                }
            }
            else if (trm is Number) {
                // 数値定数の場合

                return trm.ToString();
            }
            else if (trm is Apply) {
                // 関数適用の場合

                Apply app = trm as Apply;

                if ("+-*/%".Contains(app.Function.Name[0])) {
                    string s;

                    Debug.Assert(app.Args.Length != 1);

                    if (app.IsAdd()) {

                        s = string.Join(" ", from x in app.Args select (x == app.Args[0] || x.Value < 0 ? "" : "+ ") + TermCode(x));
                    }
                    else {

                        s = string.Join(" " + app.Function.Name + " ", from x in app.Args select TermCode(x));
                    }

                    if (app.Parent is Apply && (app.Parent as Apply).Precedence() <= app.Precedence()) {
                        return "(" + s + ")";
                    }
                    else {
                        return s;
                    }
                }
                else {

                    if (app.Function.VarRef == MkFn.Singleton.DiffFnc && app.Args[0] is Reference && (app.Args[0] as Reference).VarRef == MkFn.Singleton.EFnc) {

                        return "δ_" + app.Args[1].ToString();
                    }
                    else {

                        return app.Function.Name + "(" + string.Join(", ", from x in app.Args select TermCode(x)) + ")";
                    }
                }
            }
            else if (trm is LINQ) {
                // LINQの場合

                return LinqValue[trm as LINQ];
            }
            else {
                throw new Exception();
            }
        }

        public string VariableCode(Variable v) {
            return TypeCode(v.TypeVar) + " " + v.Name;
        }

        public string FunctionHeader(Function fnc, bool is_body) {
            StringWriter sw = new StringWriter();

            if (is_body) {

                sw.Write("{0} {1}::{2}", TypeCode(fnc.TypeVar), (fnc.ParentVar as Class).Name, fnc.Name);
            }
            else {

                sw.Write("{0} {1}", TypeCode(fnc.TypeVar), fnc.Name);
            }
            sw.Write("(" + string.Join(", ", from v in fnc.Params select VariableCode(v)) + ")");

            return sw.ToString();
        }

        public string FunctionCode(Function fnc) {
            StringWriter sw = new StringWriter();

            sw.WriteLine("");
            sw.WriteLine(FunctionHeader(fnc, true) + "{");
            sw.Write(StatementCode(fnc.BodyStatement, 0));
            sw.WriteLine("}");

            return sw.ToString();
        }

        public string FieldCode(Variable fld) {
            if(fld.Domain != null) {

                if(fld.Domain is Number) {
                    // 定義域が数値定数の場合

                    Number num = fld.Domain as Number;

                    if(fld.TypeVar == theMkFn.IntClass && num.TypeTerm == theMkFn.IntClass) {

                        return "#define " + fld.Name + " " + num.Value.ToString() + "\r\n";
                    }
                    else {
                        throw new Exception();
                    }
                }
                else if(fld.Domain.IsNew()){
                    Apply app = fld.Domain as Apply;

                    return fld.TypeVar.Name + " " + fld.Name + string.Join("", from x in app.Args select "[" + x.ToString() +"]") + ";\r\n";
                }
                else {
                    throw new Exception();
                }
            }
            else {

                return TypeCode(fld.TypeVar) + " " + fld.Name + ";\r\n";
            }
        }

        public void ClassCode(Class cls, out string header, out string body) {
            header = "struct " + cls.Name + " {\r\n" + 
                string.Join("", from fld in cls.Fields select Nest(1) + FieldCode(fld)) +
                string.Join("", from fnc in cls.Functions select Nest(1) + FunctionHeader(fnc, false) + ";\r\n") +
                "};\r\n";

            string inc = "#include \"stdafx.h\"\r\n#include \"" + cls.Name + ".h\"\r\n";
            body = inc + string.Join("\r\n", from fnc in cls.Functions select FunctionCode(fnc));
        }
    }
}
