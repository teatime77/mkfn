using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;

namespace MkFn {
    public partial class MakeCode {
        public MkFn theMkFn;
        Dictionary<LINQ, string> LinqValue = new Dictionary<LINQ, string>();
        int TmpCnt;

        /*
            コンストラクター
        */
        public MakeCode(MkFn mkfn) {
            theMkFn = mkfn;
        }

        /*
          ネストの空白を返す。  
        */
        public string Nest(int nest) {
            return new string(' ', 4 * nest);
        }

        /*
          型のコードを返す。  
        */
        public string TypeCode(Class cls) {
            Debug.Assert(cls != null);

            if(cls.DimCnt == 0) {
                // スカラーの場合

                return cls.Name;
            }
            else if (cls.DimCnt == 1) {
                // 1次元配列の場合

                return cls.Name + "*";
            }
            else {
                // 多次元配列の場合

                return string.Format("boost::multi_array<{0}, {1}>", cls.Name, cls.DimCnt);
            }
        }

        /*
          for文の先頭のコードを追加する。  
        */
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

        /*
          文のコードを返す。  
        */
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

                foreach (LINQ lnq in lnks) {
                    // 集計用の変数
                    TmpCnt++;
                    string tmp_name = string.Format("_wk{0}", TmpCnt);
                    LinqValue.Add(lnq, tmp_name);
                }

                foreach (LINQ lnq in lnks) {

                    // 作業変数の名前
                    string tmp_name = LinqValue[lnq];

                    // 作業変数を初期化する。
                    if (lnq.Aggregate.VarRef == theMkFn.SumFnc) {
                        // 和の場合

                        // 初期値 = 0
                        sw.WriteLine("{0}{1} {2} = 0;", Nest(nest), TypeCode(lnq.TypeTerm), tmp_name);
                    }
                    else if (lnq.Aggregate.VarRef == theMkFn.ProdFnc) {
                        // 積の場合

                        // 初期値 = 1
                        sw.WriteLine("{0}{1} {2} = 1;", Nest(nest), TypeCode(lnq.TypeTerm), tmp_name);
                    }
                    else if (lnq.Aggregate.VarRef == theMkFn.MaxFnc) {
                        // 最大値の場合

                        string min_const = "";
                        if(lnq.TypeTerm == theMkFn.DoubleClass) {
                            // doubleの場合

                            min_const = "-DBL_MAX";
                        }
                        else if (lnq.TypeTerm == theMkFn.FloatClass) {
                            // floatの場合
                            
                            min_const = "-FLT_MAX";
                        }
                        else if (lnq.TypeTerm == theMkFn.IntClass) {
                            // intの場合
                            
                            min_const = "INT_MIN";
                        }
                        else {
                            throw new Exception();
                        }

                        // 初期値 = 数の最小値
                        sw.WriteLine("{0}{1} {2} = {3};", Nest(nest), TypeCode(lnq.TypeTerm), tmp_name, min_const);
                    }

                    // ループ変数に対し
                    foreach (Variable loop_var in lnq.Variables) {
                        // for文の先頭のコードを追加する。  
                        ForHeadCode(loop_var, sw, nest);
                    }

                    if (lnq.Aggregate.VarRef == theMkFn.SumFnc) {
                        // 和の場合

                        // 加算する。
                        sw.WriteLine("{0}{1} += {2};", Nest(nest+1), tmp_name, TermCode(lnq.Select));
                    }
                    else if (lnq.Aggregate.VarRef == theMkFn.ProdFnc) {
                        // 積の場合

                        // 乗算する。
                        sw.WriteLine("{0}{1} *= {2};", Nest(nest+1), tmp_name, TermCode(lnq.Select));
                    }
                    else if (lnq.Aggregate.VarRef == theMkFn.MaxFnc) {
                        // 最大値の場合

                        // 最大値を更新する。
                        sw.WriteLine("{0}{1} = std::max({1}, {2});", Nest(nest+1), tmp_name, TermCode(lnq.Select));
                    }

                    // ループ変数に対し
                    foreach (Variable loop_var in lnq.Variables) {
                        sw.WriteLine(Nest(nest) + "}");
                    }
                }

                if (stmt is Assignment) {
                    // 代入文の場合

                    Assignment asn = stmt as Assignment;

                    sw.WriteLine( Nest(nest) + TermCode(asn.Left) + " = " + TermCode(asn.Right) + ";");
                }
                else {
                    // return文の場合

                    sw.WriteLine( Nest(nest) + "return " + TermCode((stmt as Return).Value) + ";");
                }
            }
            else if(stmt is ForEach) {
                // foreachの場合

                ForEach fe = stmt as ForEach;

                // for文の先頭のコードを追加する。  
                ForHeadCode(fe.LoopVariable, sw, nest);

                // forブロックの内部の文に対し
                foreach (Statement stmt2 in (stmt as BlockStatement).Statements) {
                    sw.Write(StatementCode(stmt2, nest + 1));
                }

                sw.WriteLine(Nest(nest) + "}");
            }
            else if (stmt is BlockStatement) {
                // ブロック文の場合

                foreach (Statement stmt2 in (stmt as BlockStatement).Statements) {

                    // 文のコードを追加する。
                    sw.Write(StatementCode(stmt2, nest + 1));
                }
            }
            else {
                throw new Exception();
            }

            return sw.ToString();
        }

        /*
          係数を含んだ項のコードを返す。  
        */
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

        /*
          係数を除く項の本体のコードを返す。  
        */
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
                    // 演算子の場合

                    string s;

                    Debug.Assert(app.Args.Length != 1);

                    if (app.IsAdd()) {
                        // 加算の場合

                        s = string.Join(" ", from x in app.Args select (x == app.Args[0] || x.Value < 0 ? "" : "+ ") + TermCode(x));
                    }
                    else {
                        // 加算でない場合

                        s = string.Join(" " + app.Function.Name + " ", from x in app.Args select TermCode(x));
                    }

                    if (app.Parent is Apply && (app.Parent as Apply).Precedence() <= app.Precedence()) {
                        // 親の演算子の優先順位が高い場合

                        // カッコで囲む。
                        return "(" + s + ")";
                    }
                    else {
                        return s;
                    }
                }
                else {
                    // 演算子でない場合

                    if (app.Function.VarRef == MkFn.Singleton.DiffFnc && app.Args[0] is Reference && (app.Args[0] as Reference).VarRef == MkFn.Singleton.EFnc) {

                        return "δ_" + app.Args[1].ToString();
                    }
                    else if (Term.IsNew(app)) {
                        // newの場合

                        if (app.Args.Length == 0) {
                            // スカラーの場合

                            throw new Exception();
                        }
                        else if (app.Args.Length == 1) {
                            // 1次元配列の場合

                            return app.Function.Name + " " + app.NewClass.Name + string.Join("", from x in app.Args select "[" + x.ToString() + "]");
                        }
                        else {
                            // 多次元配列の場合

                            return string.Format("boost::multi_array<{0}, {1}>(boost::extents{2});", app.NewClass.Name, app.Args.Length, string.Join("", from x in app.Args select "[" + x.ToString() + "]"));
                        }

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

        /*
          変数の宣言のコードを返す。  
        */
        public string VariableCode(Variable v) {
            return TypeCode(v.TypeVar) + " " + v.Name;
        }

        /*
          関数のヘッダーのコードを返す。  
        */
        public string FunctionHeader(Class cls, Function fnc, bool is_body) {
            StringWriter sw = new StringWriter();
            bool is_constructor = (fnc.Name == MkFn.ConstructorName(cls));

            if (is_body) {
                // 関数の本体の場合

                if (is_constructor) {
                    // コンストラクターの場合

                    sw.Write("{0}::{1}", (fnc.ParentVar as Class).Name, fnc.Name);
                }
                else {
                    // コンストラクターでない場合

                    sw.Write("{0} {1}::{2}", TypeCode(fnc.TypeVar), (fnc.ParentVar as Class).Name, fnc.Name);
                }
            }
            else {
                // 関数の宣言の場合

                if (is_constructor) {
                    // コンストラクターの場合

                    sw.Write("{0}", fnc.Name);
                }
                else {
                    // コンストラクターでない場合

                    sw.Write("{0} {1}", TypeCode(fnc.TypeVar), fnc.Name);
                }
            }
            sw.Write("(" + string.Join(", ", from v in fnc.Params select VariableCode(v)) + ")");

            return sw.ToString();
        }

        /*
          関数の定義のコードを返す。  
        */
        public string FunctionCode(Class cls, Function fnc) {
            StringWriter sw = new StringWriter();
            TmpCnt = 0;

            sw.WriteLine("");
            sw.WriteLine(FunctionHeader(cls, fnc, true) + "{");
            sw.Write(StatementCode(fnc.BodyStatement, 0));
            sw.WriteLine("}");

            return sw.ToString();
        }

        /*
          フィールドの宣言のコードを返す。  
        */
        public string FieldCode(Variable fld) {
            return TypeCode(fld.TypeVar) + " " + fld.Name + ";\r\n";
        }

        /*
          クラスのコードを返す。 
        */
        public void ClassCode(Class cls, out string header, out string body) {
            header = "struct " + cls.Name + " {\r\n" + 
                string.Join("", from fld in cls.Fields select Nest(1) + FieldCode(fld)) +
                string.Join("", from fnc in cls.Functions select Nest(1) + FunctionHeader(cls, fnc, false) + ";\r\n") +
                "};\r\n";

            string inc = "#include \"stdafx.h\"\r\n#include \"boost/multi_array.hpp\"\r\n#include \"MkFn.h\"\r\n#include \"" + cls.Name + ".h\"\r\n";
            body = inc + string.Join("\r\n", from fnc in cls.Functions select FunctionCode(cls, fnc));
        }
    }
}
