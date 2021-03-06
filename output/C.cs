﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;

namespace MkFn {
    public partial class MkFn {
        public int TmpCnt;

        /*
          ネストの空白を返します。  
        */
        public string Nest(int nest) {
            return new string(' ', 4 * nest);
        }

        /*
          for文の先頭のコードを追加します。  
        */
        public void ForHeadCode(Variable loop_var, StringWriter sw, int nest) {
            if (!IsRange(loop_var.Domain)) {
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
          文のコードを返します。  
        */
        public string StatementCode(Statement stmt, int nest) {
            StringWriter sw = new StringWriter();

            if (stmt is Assignment || stmt is Return) {
                // 代入文かreturnの場合

                sw.WriteLine("");

                Dictionary<LINQ, string> linq_value = LinqValue;
                LinqValue = null;
                Language lang_save = OutputLanguage;
                OutputLanguage = Language.CS;
                sw.WriteLine(Nest(nest) + "// " + stmt.ToString());

                LinqValue = linq_value;
                OutputLanguage = lang_save;

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

                foreach (LINQ lnq in lnks) {

                    // 作業変数の名前
                    string tmp_name = MkFn.LinqValue[lnq];

                    // 作業変数を初期化します。
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

                        if (! (lnq.Select is Reference)) {
                            // 変数参照でない場合

                            throw new Exception();
                        }

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
                        sw.WriteLine("{0}int {1}_max_idx = 0;", Nest(nest), tmp_name);
                    }

                    // ループの内部のネスト
                    int inner_nest = nest + lnq.Variables.Length;

                    // ループ変数に対し
                    for (int i = 0; i < lnq.Variables.Length; i++) {
                        // for文の先頭のコードを追加します。  
                        ForHeadCode(lnq.Variables[i], sw, nest + i);
                    }

                    if (lnq.Aggregate.VarRef == SumFnc) {
                        // 和の場合

                        // 加算します。
                        sw.WriteLine("{0}{1} += {2};", Nest(inner_nest), tmp_name, lnq.Select.ToString());
                    }
                    else if (lnq.Aggregate.VarRef == ProdFnc) {
                        // 積の場合

                        // 乗算します。
                        sw.WriteLine("{0}{1} *= {2};", Nest(inner_nest), tmp_name, lnq.Select.ToString());
                    }
                    else if (lnq.Aggregate.VarRef == MaxFnc) {
                        // 最大値の場合

                        Reference max_ref = lnq.Select as Reference;

                        string idx = OffsetFromIndexes(max_ref);

                        // 最大値を更新します。
                        sw.WriteLine("{0}int {1}_idx = {2};", Nest(inner_nest), tmp_name, idx);
                        sw.WriteLine("{0}if({1} < {2}[{1}_idx]) {{", Nest(inner_nest), tmp_name, max_ref.Name);
                        sw.WriteLine("{0}{1} = {2}[{1}_idx];", Nest(inner_nest + 1), tmp_name, max_ref.Name);
                        sw.WriteLine("{0}{1}_max_idx = {1}_idx;", Nest(inner_nest + 1), tmp_name);
                        sw.WriteLine("{0}}}", Nest(inner_nest));
                    }
                    
                    // ループ変数に対し
                    for (int i = lnq.Variables.Length - 1; 0 <= i; i--) {
                        sw.WriteLine(Nest(nest + i) + "}");
                    }

                    if (lnq.Aggregate.VarRef == MaxFnc) {
                        // 最大値の場合

                        Assignment asn = stmt as Assignment;

                        Variable max_var = (lnq.Select as Reference).VarRef;

                        string idx = OffsetFromIndexes(asn.Left);

                        sw.WriteLine("{0}{1}[{2}] = {3}_max_idx;", Nest(nest), IndexName(max_var), idx, tmp_name);
                    }

                }

                if (stmt is Assignment) {
                    // 代入文の場合

                    Assignment asn = stmt as Assignment;

                    string asn_opr;
                    if(asn.Left.VarRef == delta_y_var) {

                        asn_opr = " += ";
                    }
                    else {

                        asn_opr = " = ";
                    }

                    sw.WriteLine( Nest(nest) + asn.Left.ToString() + asn_opr + asn.Right.ToString() + ";");
                }
                else {
                    // return文の場合

                    sw.WriteLine( Nest(nest) + "return " + (stmt as Return).Value.ToString() + ";");
                }
            }
            else if(stmt is ForEach) {
                // foreachの場合

                ForEach fe = stmt as ForEach;

                // for文の先頭のコードを追加します。  
                for(int i = 0; i < fe.LoopVariables.Count; i++) {
                    ForHeadCode(fe.LoopVariables[i], sw, nest + i);
                }

                // forブロックの内部の文に対し
                foreach (Statement stmt2 in (stmt as BlockStatement).Statements) {
                    sw.Write(StatementCode(stmt2, nest + fe.LoopVariables.Count));
                }

                for (int i = 0; i < fe.LoopVariables.Count; i++) {
                    sw.WriteLine(Nest(nest + i) + "}");
                }
            }
            else if (stmt is BlockStatement) {
                // ブロック文の場合

                foreach (Statement stmt2 in (stmt as BlockStatement).Statements) {

                    // 文のコードを追加します。
                    sw.Write(StatementCode(stmt2, nest + 1));
                }
            }
            else {
                throw new Exception();
            }

            return sw.ToString();
        }

        /*
          変数の宣言のコードを返します。  
        */
        public string VariableCode(Variable v) {
            return v.TypeVar.ToString() + " " + v.Name;
        }

        /*
          関数のヘッダーのコードを返します。  
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
          フィールドの宣言のコードを返します。  
        */
        public string FieldCode(Variable fld) {
            return fld.TypeVar.ToString() + " " + fld.Name + ";\r\n";
        }
    }
}
