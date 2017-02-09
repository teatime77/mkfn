﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Diagnostics;

namespace mkfn {
    /*
        クラス
    */
    public class Class {
        public Class Parent;
        public string Name;
        public List<Variable> Fields = new List<Variable>();
        public List<Function> Functions = new List<Function>();

        public Class(string name) {
            Name = name;
        }

        public virtual Class ValueType() {
            return this;
        }

        public override string ToString() {
            return Name;
        }

        public bool IsSubClass(Class cls) {
            return Parent == cls;
        }
    }

    /*
        配列の型
    */
    public class ArrayType : Class {
        public Class ElementType;
        public int DimCnt;

        public ArrayType(Class element_type, int dim_cnt) : base(element_type.Name) {
            ElementType = element_type;
            DimCnt = dim_cnt;
        }

        public override Class ValueType() {
            return ElementType;
        }

        public override string ToString() {
            return Name + "[" + new string(',', DimCnt - 1) + "]";
        }
    }

    /*
        変数
    */
    public class Variable {
        public object ParentVar;
        public string Name;
        public Class TypeVar;
        public Term Domain;

        public Variable(string name, Class type, Term domain) {
            Name = name;
            TypeVar = type;
            Domain = domain;

            if (Domain != null) {
                Domain.Parent = this;
            }
        }

        public Variable Clone(Dictionary<Variable, Variable> var_tbl) {
            Term domain = (Domain == null ? null : Domain.Clone(var_tbl));
            Variable v1 = new Variable(Name, TypeVar, domain);
            var_tbl.Add(this, v1);

            return v1;
        }

        public override string ToString() {
            if (TypeVar == null) {
                return Name;
            }
            return TypeVar.ToString() + " " + Name;
        }
    }

    /*
        関数
    */
    public class Function : Variable {
        public List<Variable> Params = new List<Variable>();

        public BlockStatement Statement;
        public Function(string name, Class type) : base(name, type, null) {
        }
    }

    /*
        文
    */
    public class Statement {
        public object ParentStmt;
    }

    /*
        代入文
    */
    public class Assignment : Statement {
        public Reference Left;
        public Term Right;

        public Assignment(Reference left, Term right) {
            Left = left;
            Right = right;

            Left.Parent = this;
            Right.Parent = this;
        }

        public override string ToString() {
            return Left.ToString() + " = " + Right.ToString();
        }
    }

    /*
        return文
    */
    public class Return : Statement {
        public Term Value;

        public Return(Term value) {
            Value = value;
        }
    }

    /*
        ブロック文
    */
    public class BlockStatement : Statement {
        public List<Statement> Statements;

        public BlockStatement(List<Statement> statements) {
            Statements = statements;

            foreach (Statement s in Statements) {
                s.ParentStmt = this;
            }
        }
    }

    /*
        foreach文
    */
    public class ForEach : BlockStatement {
        public Variable LoopVariable;

        public ForEach(Variable variable, List<Statement> statements) : base(statements) {
            LoopVariable = variable;

            LoopVariable.ParentVar = this;
            foreach (Statement s in Statements) {
                s.ParentStmt = this;
            }
        }
    }

    /*
        項
    */
    public abstract class Term {
        public object Parent;
        public double Value = 1;

        public abstract bool EqBody(Object obj);

        public virtual bool Eq(Object obj) {
            return obj is Term && Value == (obj as Term).Value && EqBody(obj);
        }

        public Term Clone(Dictionary<Variable, Variable> var_tbl = null) {
            if (var_tbl == null) {
                var_tbl = new Dictionary<Variable, Variable>();
            }

            if (this is Reference) {
                return (this as Reference).Clone(var_tbl);
            }
            else if (this is Number) {
                return (this as Number).Clone(var_tbl);
            }
            else if (this is Apply) {
                return ToApply().Clone(var_tbl);
            }
            else if (this is LINQ) {
                return (this as LINQ).Clone(var_tbl);
            }
            else {
                Debug.Assert(false);
                return null;
            }
        }

        public Term Minus() {
            Value *= -1;

            return this;
        }

        public Number ToNumber() {
            return this as Number;
        }

        public Apply ToApply() {
            return this as Apply;
        }

        public Reference AsReference() {
            return this as Reference;
        }

        public bool IsAdd() {
            return this is Apply && ToApply().Function.VarRef == mkfn.Singleton.AddFnc;
        }

        public bool IsMul() {
            return this is Apply && ToApply().Function.VarRef == mkfn.Singleton.MulFnc;
        }

        public override string ToString() {
            if (!(this is Number)) {

                if (Value == 1) {

                    return ToStringBody();
                }
                else if (Value == -1) {

                    return "- " + ToStringBody();
                }
                else {

                    return Value.ToString() + " * " + ToStringBody();
                }
            }
            else {

                return ToStringBody();
            }
        }

        public abstract string ToStringBody();

        public List<Reference> AllRefs() {
            List<Reference> all_refs = new List<Reference>();

            mkfn.Navi(this,
                delegate (object obj) {
                    if (obj is Reference) {
                        all_refs.Add(obj as Reference);
                    }
                });

            return all_refs;
        }
    }

    /*
        数値定数
    */
    public class Number : Term {
        public Number(string text) {

            if (!double.TryParse(text, out Value)) {
                throw new SyntaxException();
            }
        }

        public Number(double d) {
            Value = d;
        }

        public new Number Clone(Dictionary<Variable, Variable> var_tbl) {
            return new Number(Value);
        }

        public override bool EqBody(Object obj) {
            return obj is Number;
        }

        public override string ToStringBody() {
            return Value.ToString();
        }
    }

    /*
        変数参照
    */
    public class Reference : Term {
        public string Name;
        public Variable VarRef;
        public Term[] Indexes;

        public Reference(string name, Variable ref_var, Term[] idx, double val = 1) {
            Name = name;
            VarRef = ref_var;
            Indexes = idx;
            Value = val;

            if (Indexes != null) {
                foreach (Term t in Indexes) {
                    t.Parent = this;
                }
            }
        }

        public Reference(Variable v, double val = 1) {
            Name = v.Name;
            VarRef = v;
            Indexes = null;
        }

        public new Reference Clone(Dictionary<Variable, Variable> var_tbl = null) {
            if (var_tbl == null) {
                var_tbl = new Dictionary<Variable, Variable>();
            }
            Variable v1;
            if (!var_tbl.TryGetValue(VarRef, out v1)) {
                v1 = VarRef;
            }

            if (Indexes == null) {
                return new Reference(Name, v1, null, Value);
            }

            Term[] idx = (from t in Indexes select t.Clone(var_tbl)).ToArray();

            return new Reference(Name, v1, idx, Value);
        }

        public override bool EqBody(Object obj) {
            if (!(obj is Reference)) {
                return false;
            }
            Reference r = obj as Reference;
            if (r.VarRef != VarRef) {
                return false;
            }
            if ((Indexes == null) != (r.Indexes == null)) {
                return false;
            }
            if (Indexes == null) {
                return true;
            }
            else {
                Debug.Assert(Indexes.Length == r.Indexes.Length);
                for (int i = 0; i < Indexes.Length; i++) {
                    if (!Indexes[i].Eq(r.Indexes[i])) {
                        return false;
                    }
                }
                return true;
            }
        }

        public bool Defined() {
            return Parent is Assignment && (Parent as Assignment).Left == this;
        }

        public bool Used() {
            return !Defined();
        }

        public override string ToStringBody() {
            if (Indexes == null) {
                return Name;
            }
            else {
                return Name + "[" + string.Join(", ", from x in Indexes select x.ToString()) + "]";
            }
        }
    }

    /*
        関数適用
    */
    public class Apply : Term {
        public Reference Function;
        public Term[] Args;

        public Apply(Reference function, Term[] args) {
            Function = function;
            Args = args;

            Function.Parent = this;
            foreach (Term t in Args) {
                t.Parent = this;
            }
        }

        public Apply(Variable function, params Term[] args) : this(new Reference(function), args) {
        }

        public new Apply Clone(Dictionary<Variable, Variable> var_tbl = null) {
            Term[] args = (from t in Args select t.Clone(var_tbl)).ToArray();
            Apply app = new Apply(Function.Clone(var_tbl), args);
            app.Value = Value;

            return app;
        }

        public override string ToStringBody() {
            if ("+-*/%".Contains(Function.Name[0])) {
                string s;

                Debug.Assert(Args.Length != 1);

                if (IsAdd()) {

                    //s = string.Join(" ", from x in Args select (x == Args[0] || x.Value < 0 ? "" : "+ ") + (Math.Abs(x.Value) == 1 ? x.ToStringBody() : x.ToString()));
                    s = string.Join(" ", from x in Args select (x == Args[0] || x.Value < 0 ? "" : "+ ") + x.ToString());
                }
                else {

                    s = string.Join(" " + Function.Name + " ", from x in Args select x.ToString());
                }

                if (Parent is Apply && (Parent as Apply).Precedence() <= Precedence()) {
                    return "(" + s + ")";
                }
                else {
                    return s;
                }
            }
            else {

                if (Function.VarRef == mkfn.Singleton.DiffFnc && Args[0] is Reference && (Args[0] as Reference).VarRef == mkfn.Singleton.EFnc) {

                    return "δ_" + Args[1].ToString();
                }
                else {

                    return Function.Name + "(" + string.Join(", ", from x in Args select x.ToString()) + ")";
                }
            }
        }

        public override bool EqBody(Object obj) {
            if (!(obj is Apply)) {
                return false;
            }

            Apply app = obj as Apply;

            if ( ! Function.Eq(app.Function) ) {
                return false;
            }
            if (Args.Length != app.Args.Length) {
                return false;
            }

            for (int i = 0; i < Args.Length; i++) {
                if (!Args[i].Eq(app.Args[i])) {
                    return false;
                }
            }
            return true;
        }

        public int Precedence() {
            if (Char.IsLetter(Function.Name[0])) {
                return 20;
            }

            if ("*/%".Contains(Function.Name[0])) {
                return 1;
            }

            if (Args.Length == 1 && "+-!".Contains(Function.Name[0])) {
                return 2;
            }

            return 10;
        }
    }

    /*
        LINQ
    */
    public class LINQ : Term {
        public Variable[] Variables;
        public Term Select;
        public Reference Aggregate;

        public LINQ(Variable[] variables, Term select, Reference aggregate, double val = 1) {
            Variables = variables;
            Select = select;
            Aggregate = aggregate;
            Value = val;

            foreach (Variable v in Variables) {
                v.ParentVar = this;
            }
            Select.Parent = this;
            if (Aggregate != null) {

                Aggregate.Parent = this;
            }
        }

        public override bool EqBody(Object obj) {
            return this == obj;
        }

        public new LINQ Clone(Dictionary<Variable, Variable> var_tbl) {
            Variable[] vars = (from v in Variables select v.Clone(var_tbl)).ToArray();
            return new LINQ(vars, Select.Clone(var_tbl), (Aggregate == null ? null : Aggregate.Clone(var_tbl)), Value);
        }

        public override string ToStringBody() {
            string list = string.Join(" ", from x in Variables select "from " + x.Name + " in " + x.Domain.ToString()) + " select " + Select.ToString();
            if (Aggregate == null) {

                return list;
            }
            return "(" + list + ")." + Aggregate.Name + "()";
        }
    }

    /*
        構文エラー
    */
    public class SyntaxException : Exception {

        public SyntaxException() {
            Token tkn = mkfn.Singleton.CurrentToken;
            string line = mkfn.Singleton.Lines[tkn.LineIndex];
            Debug.WriteLine("Err {0}行: {1} ^ {2}", tkn.LineIndex + 1, line.Substring(0, tkn.CharPos), line.Substring(tkn.CharPos));
            Debug.Write("");
        }
    }
}
