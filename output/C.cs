using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;

namespace MkFn {
    public partial class MkFn {
        public int TmpCnt;

        /*
          ネストの空白を返す。  
        */
        public string Nest(int nest) {
            return new string(' ', 4 * nest);
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

                sw.WriteLine("{0}for(int {1} = 0; {1} < {2}; {1}++){{", Nest(nest), loop_var.Name, rng.Args[0].ToString());
            }
            else if (rng.Args.Length == 2) {

                sw.WriteLine("{0}for(int {1} = {2}; {1} < {3}; {1}++){{", Nest(nest), loop_var.Name, rng.Args[0].ToString(), rng.Args[1].ToString());
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

                // すべてのLINQのリスト
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
                    MkFn.LinqValue.Add(lnq, tmp_name);
                }

                sw.WriteLine("");
                sw.WriteLine(Nest(nest) + "// " + stmt.ToString());

                foreach (LINQ lnq in lnks) {

                    // 作業変数の名前
                    string tmp_name = MkFn.LinqValue[lnq];

                    // 作業変数を初期化する。
                    if (lnq.Aggregate.VarRef == SumFnc) {
                        // 和の場合

                        // 初期値 = 0
                        sw.WriteLine("{0}{1} {2} = 0;", Nest(nest), lnq.TypeTerm.ToString(), tmp_name);
                    }
                    else if (lnq.Aggregate.VarRef == ProdFnc) {
                        // 積の場合

                        // 初期値 = 1
                        sw.WriteLine("{0}{1} {2} = 1;", Nest(nest), lnq.TypeTerm.ToString(), tmp_name);
                    }
                    else if (lnq.Aggregate.VarRef == MaxFnc) {
                        // 最大値の場合

                        string min_const = "";
                        if(lnq.TypeTerm == DoubleClass) {
                            // doubleの場合

                            min_const = "-DBL_MAX";
                        }
                        else if (lnq.TypeTerm == FloatClass) {
                            // floatの場合
                            
                            min_const = "-FLT_MAX";
                        }
                        else if (lnq.TypeTerm == IntClass) {
                            // intの場合
                            
                            min_const = "INT_MIN";
                        }
                        else {
                            throw new Exception();
                        }

                        // 初期値 = 数の最小値
                        sw.WriteLine("{0}{1} {2} = {3};", Nest(nest), lnq.TypeTerm.ToString(), tmp_name, min_const);
                    }

                    // ループ変数に対し
                    foreach (Variable loop_var in lnq.Variables) {
                        // for文の先頭のコードを追加する。  
                        ForHeadCode(loop_var, sw, nest);
                    }

                    if (lnq.Aggregate.VarRef == SumFnc) {
                        // 和の場合

                        // 加算する。
                        sw.WriteLine("{0}{1} += {2};", Nest(nest+1), tmp_name, lnq.Select.ToString());
                    }
                    else if (lnq.Aggregate.VarRef == ProdFnc) {
                        // 積の場合

                        // 乗算する。
                        sw.WriteLine("{0}{1} *= {2};", Nest(nest+1), tmp_name, lnq.Select.ToString());
                    }
                    else if (lnq.Aggregate.VarRef == MaxFnc) {
                        // 最大値の場合

                        // 最大値を更新する。
                        sw.WriteLine("{0}{1} = std::max({1}, {2});", Nest(nest+1), tmp_name, lnq.Select.ToString());
                    }

                    // ループ変数に対し
                    foreach (Variable loop_var in lnq.Variables) {
                        sw.WriteLine(Nest(nest) + "}");
                    }
                }

                if (stmt is Assignment) {
                    // 代入文の場合

                    Assignment asn = stmt as Assignment;

                    sw.WriteLine( Nest(nest) + asn.Left.ToString() + " = " + asn.Right.ToString() + ";");
                }
                else {
                    // return文の場合

                    sw.WriteLine( Nest(nest) + "return " + (stmt as Return).Value.ToString() + ";");
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
          変数の宣言のコードを返す。  
        */
        public string VariableCode(Variable v) {
            return v.TypeVar.ToString() + " " + v.Name;
        }

        /*
          関数のヘッダーのコードを返す。  
        */
        public string FunctionHeader(Class cls, Function fnc, bool is_body) {
            StringWriter sw = new StringWriter();
            bool is_constructor = fnc.IsConstructor();

            if (is_body) {
                // 関数の本体の場合

                if (is_constructor) {
                    // コンストラクターの場合

                    sw.Write("{0}::{1}", (fnc.ParentVar as Class).Name, fnc.Name);
                }
                else {
                    // コンストラクターでない場合

                    sw.Write("{0} {1}::{2}", fnc.TypeVar.ToString(), (fnc.ParentVar as Class).Name, fnc.Name);
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

                    sw.Write("{0} {1}", fnc.TypeVar.ToString(), fnc.Name);
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
            return fld.TypeVar.ToString() + " " + fld.Name + ";\r\n";
        }

        /*
          クラスのコードを返す。 
        */
        public void ClassCode(Class cls, out string header, out string body) {
            header = "struct " + cls.Name + " {\r\n" + 
                string.Join("", from fld in cls.Fields select Nest(1) + FieldCode(fld)) +
                string.Join("", from fnc in cls.Functions select Nest(1) + FunctionHeader(cls, fnc, false) + ";\r\n") +
                "};\r\n";

            string inc = "#include \"boost/multi_array.hpp\"\r\n#include \"MkFn.h\"\r\n#include \"" + cls.Name + ".h\"\r\n";
            body = inc + string.Join("\r\n", from fnc in cls.Functions select FunctionCode(cls, fnc));
        }
    }
}
