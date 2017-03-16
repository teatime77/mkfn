using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Diagnostics;
using System.Reflection;

namespace MkFn {
    public partial class MkFn {
        public static MkFn Singleton;

        string HomeDir = @"C:\usr\prj\mkfn\mkfn";
        public string[] Lines;
        List<Class> SimpleTypes = new List<Class>();
        List<ArrayType> ArrayTypes = new List<ArrayType>();
        List<Class> AppClasses = new List<Class>();

        public int TokenPos;
        public Token CurrentToken;
        Token NextToken;
        
        List<Token> TokenList;
        Token EOTToken = new Token(TokenType.EOT, TokenSubType.Unknown, null, -1, -1);

        Token ReadNextToken() {
            Token current_token = CurrentToken;

            CurrentToken = NextToken;

            // トークンの位置を1つ進める。
            TokenPos++;

            if (TokenPos + 1 < TokenList.Count) {

                NextToken = TokenList[TokenPos + 1];
            }
            else{

                NextToken = EOTToken;
            }

            return current_token;
        }

        Token GetToken(string text) {
            if(CurrentToken.Text != text) {

                throw new SyntaxException();
            }

            return ReadNextToken();
        }

        Token GetToken(TokenType type) {
            if (type != TokenType.Any && type != CurrentToken.Type) {

                throw new SyntaxException();
            }

            return ReadNextToken();
        }

        void ReadUsing() {
            GetToken("using");
            GetToken(TokenType.Identifier);

            while (CurrentToken.Text == ".") {

                GetToken(".");
                GetToken(TokenType.Identifier);
            }

            GetToken(";");

        }

        void ReadModifier() {
            while (true) {
                if(CurrentToken.Text == "public") {
                    GetToken(TokenType.Any);
                }
                else if (CurrentToken.Text == "abstract") {
                    GetToken(TokenType.Any);
                }
                else if (CurrentToken.Text == "static") {
                    GetToken(TokenType.Any);
                }
                else if (CurrentToken.Text == "override") {
                    GetToken(TokenType.Any);
                }
                else if (CurrentToken.Text == "const") {
                    GetToken(TokenType.Any);
                }
                else {
                    break;
                }
            }
        }

        Class GetSimpleType(string name) {

            var v = from c in SimpleTypes where c.Name == name select c;
            if (v.Any()) {

                return v.First();
            }

            Class type = new Class(name);
            SimpleTypes.Add(type);

            return type;
        }

        Class ReadType() {
            Token type_id = GetToken(TokenType.Identifier);

            if (CurrentToken.Text != "[") {
                // 配列でない場合

                return GetSimpleType(type_id.Text);
            }
            else { 
                // 配列の場合


                GetToken("[");

                int dim_cnt = 1;

                while (CurrentToken.Text == ",") {
                    dim_cnt++;
                    GetToken(",");
                }

                GetToken("]");

                var v = from c in ArrayTypes where c.Name == type_id.Text && c.DimCnt == dim_cnt select c;
                if (v.Any()) {

                    return v.First();
                }

                ArrayType type = new ArrayType(GetSimpleType(type_id.Text), dim_cnt);
                ArrayTypes.Add(type);

                return type;
            }
        }

        Variable ReadVariable() {
            Class type = ReadType();
            Token id = GetToken(TokenType.Identifier);

            return new Variable(id.Text, type, null);
        }

        Reference ReadReference() {
            Token id = GetToken(TokenType.Identifier);

            if(CurrentToken.Text != "[") {

                return new Reference(id.Text, null, null);
            }
            else {
                GetToken("[");

                List<Term> idxes = new List<Term>();
                while (true) {
                    Term idx = Expression();
                    idxes.Add(idx);

                    if (CurrentToken.Text == "]") {
                        break;
                    }
                    GetToken(",");
                }
                GetToken("]");

                return new Reference(id.Text, null, idxes.ToArray());
            }
        }

        List<Term> ReadArgs() {
            List<Term> terms = new List<Term>();
            GetToken("(");

            if (CurrentToken.Text != ")") {

                while (true) {
                    terms.Add(Expression());

                    if (CurrentToken.Text == ")") {

                        break;
                    }
                    GetToken(",");
                }
            }
            GetToken(")");

            return terms;

        }

        public Term PrimaryExpression() {
            if(CurrentToken.Type == TokenType.Identifier) {
                Reference r = ReadReference();

                if(CurrentToken.Text != "(") {
                    return r;
                }

                List<Term> args = ReadArgs();

                if(r.Name == "Prod") {

                    if(args.Count == 1 && args[0] is LINQ) {
                        // 引数がLINQの場合

                        LINQ lnq = args[0] as LINQ;
                        lnq.Aggregate = new Reference(ProdFnc);
                        lnq.Aggregate.Parent = lnq;

                        return lnq;
                    }
                    else {
                        throw new SyntaxException();
                    }
                }

                return new Apply(r, args.ToArray());
            }
            else if (CurrentToken.Type == TokenType.Number) {
                Token num = GetToken(TokenType.Number);

                return new Number(num.Text, num.SubType);
            }
            else if (CurrentToken.Text == "(") {

                GetToken("(");
                Term term = Expression();
                GetToken(")");

                if(CurrentToken.Text == ".") {

                    if (!(term is LINQ)) {
                        // LINQでない場合

                        throw new SyntaxException();
                    }

                    LINQ lnq = term as LINQ;

                    GetToken(".");
                    Token id = GetToken(TokenType.Identifier);

                    GetToken("(");
                    GetToken(")");

                    if(id.Text == SumFnc.Name) {

                        lnq.Aggregate = new Reference(SumFnc);
                    }
                    else if (id.Text == MaxFnc.Name) {

                        lnq.Aggregate = new Reference(MaxFnc);
                    }
                    else {

                        throw new SyntaxException();
                    }
                    lnq.Aggregate.Parent = lnq;

                    return lnq;
                }

                return term;
            }
            else if (CurrentToken.Text == "from") {
                List<Variable> vars = new List<Variable>();

                while (CurrentToken.Text == "from") {

                    GetToken("from");
                    Token id = GetToken(TokenType.Identifier);
                    GetToken("in");
                    Term domain = Expression();

                    vars.Add(new Variable(id.Text, null, domain));
                }

                GetToken("select");

                Term term = Expression();

                return new LINQ(vars.ToArray(), term, null);
            }
            else if (CurrentToken.Text == "new") {
                List<Term> args = new List<Term>();

                GetToken("new");

                Token type_id = GetToken(TokenType.Identifier);
                Class cls = GetSimpleType(type_id.Text);

                GetToken("[");
                while (true) {

                    args.Add( Expression() );
                    if(CurrentToken.Text != ",") {
                        break;
                    }
                    GetToken(",");
                }
                GetToken("]");

                return new Apply(MkFn.Singleton.NewFnc, cls, args.ToArray());
            }
            else {
                throw new SyntaxException();
            }
        }


        public Term UnaryExpression() {
            if (CurrentToken.Text == "-") {

                GetToken("-");

                Term t1 = PrimaryExpression();

                // 符号を反転する。
                t1.Value *= -1;

                return t1;
            }
            else {

                return PrimaryExpression();
            }
        }

        public Term MultiplicativeExpression() {
            Term t1 = UnaryExpression();

            while (CurrentToken.Text == "*" || CurrentToken.Text == "/") {

                string opr = CurrentToken.Text;

                List<Term> args = new List<Term>();
                args.Add(t1);

                while (CurrentToken.Text == opr) {
                    GetToken(opr);
                    args.Add(UnaryExpression());
                }

                if (opr == "*") {

                    t1 = new Apply(new Reference(MulFnc), args.ToArray());
                }
                else {

                    t1 = new Apply(new Reference(DivFnc), args.ToArray());
                }
            }

            return t1;
        }

        public Term AdditiveExpression() {
            Term t1 = MultiplicativeExpression();

            while (CurrentToken.Text == "+" || CurrentToken.Text == "-") {

                string opr = CurrentToken.Text;

                List<Term> args = new List<Term>();
                args.Add(t1);


                while(CurrentToken.Text == opr) {
                    GetToken(opr);
                    args.Add( MultiplicativeExpression() );
                }

                if(opr == "-") {
                    // 減算の場合

                    // 2番目以降の項の符号を反転する。
                    for (int i = 1; i < args.Count; i++) {

                        args[i].Value *= -1;
                    }
                }
                t1 = new Apply(new Reference(AddFnc), args.ToArray());
            }

            return t1;
        }


        Term Expression() {
            return AdditiveExpression();
        }

        List<Statement> ReadBlockStatement() {
            List<Statement> statements = new List<Statement>();
            GetToken("{");

            while(CurrentToken.Text != "}") {
                statements.Add( ReadStatement() );
            }
            GetToken("}");

            return statements;
        }

        Statement ReadStatement() {
            if(CurrentToken.Type == TokenType.Identifier) {
                Reference left = ReadReference();

                GetToken("=");

                Term right = Expression();
                GetToken(";");

                Assignment asn = new Assignment(left, right);
                //Debug.WriteLine(asn.ToString());
                return asn;
            }
            else if(CurrentToken.Text == "return") {

                GetToken("return");
                Term value = Expression();
                GetToken(";");

                return new Return(value);
            }
            else if(CurrentToken.Text == "foreach") {

                GetToken("foreach");
                GetToken("(");
                Variable v = ReadVariable();
                GetToken("in");
                v.Domain = Expression();
                v.Domain.Parent = v;
                GetToken(")");

                List<Statement> stmts = ReadBlockStatement();

                return new ForEach(v, stmts);
            }
            else {
                throw new SyntaxException();
            }
        }

        Function ReadFunction(string name, Class type) {
            Function fnc = new Function(name, type);

            GetToken("(");

            if (CurrentToken.Text != ")") {

                while (true) {
                    Variable arg = ReadVariable();

                    fnc.Params.Add(arg);

                    if (CurrentToken.Text == ")") {

                        break;
                    }
                    GetToken(",");
                }
            }
            GetToken(")");

            if (CurrentToken.Text == ";") {

                GetToken(";");
            }
            else {

                fnc.BodyStatement = new BlockStatement(ReadBlockStatement());
                fnc.BodyStatement.ParentStmt = fnc;
            }

            return fnc;
        }

        Class ReadClass() {
            ReadModifier();

            GetToken("class");
            Token class_id = GetToken(TokenType.Identifier);
            Class cls = GetSimpleType(class_id.Text);
            
            if (CurrentToken.Text == ":") {

                GetToken(":");
                Token super_class = GetToken(TokenType.Identifier);

                cls.Parent = GetSimpleType(super_class.Text);
            }
            GetToken("{");

            while (CurrentToken.Text != "}") {

                ReadModifier();
                Class type = ReadType();

                if (CurrentToken.Text == "(") {
                    // コンストラクターの場合

                    Function fnc = ReadFunction(MkFn.ConstructorName(cls), type);
                    cls.Functions.Add(fnc);
                    fnc.ParentVar = cls;
                }
                else {
                    // コンストラクターでない場合

                    Token id = GetToken(TokenType.Identifier);

                    if (CurrentToken.Text == ";" || CurrentToken.Text == "=") {
                        // フィールドの場合

                        Term init = null;

                        if (CurrentToken.Text == "=") {

                            GetToken("=");

                            init = Expression();
                        }

                        GetToken(";");

                        Variable field = new Variable(id.Text, type, init);

                        cls.Fields.Add(field);
                        field.ParentVar = cls;

                        //Debug.WriteLine("field {0}", field.ToString(), "");
                    }
                    else {
                        // メソッドの場合

                        Function fnc = ReadFunction(id.Text, type);
                        cls.Functions.Add(fnc);
                        fnc.ParentVar = cls;
                    }
                }
            }

            GetToken("}");

            return cls;
        }

        void ParseSource() {
            TokenPos = 0;
            CurrentToken = TokenList[0];
            NextToken = TokenList[1];

            while (CurrentToken.Text == "using") {

                ReadUsing();
            }

            while (CurrentToken != EOTToken) {

                Class cls = ReadClass();
                AppClasses.Add(cls);
            }
        }

        public void ParseSourceFile(string path) {
            Lines = File.ReadAllLines(path, Encoding.UTF8);

            TokenList = new List<Token>();
            for (int i = 0; i < Lines.Length; i++) {
                TokenList.AddRange(RemoveComment( LexicalAnalysis(Lines[i], i) ));
            }

            ParseSource();
        }

        public void Main() {
            HomeDir = Path.GetDirectoryName(Path.GetDirectoryName(Path.GetDirectoryName( Assembly.GetExecutingAssembly().Location ) ) );
            ParseSourceFile(HomeDir + @"\NeuralNetwork.cs");


            Class layer = (from cls in AppClasses where cls.Name == "Layer" select cls).First();

            EFnc = (from fld in layer.Fields where fld.Name == "E" select fld).First();

            RangeFnc = (from f in layer.Functions where f.Name == "Range" select f).First();

            σ_prime = (from fnc in layer.Functions where fnc.Name == "σ_prime" select fnc).First();
            tanh_prime = (from fnc in layer.Functions where fnc.Name == "tanh_prime" select fnc).First();

            Layers = (from cls in AppClasses where cls.IsSubClass(layer) select cls).ToArray();

            // 名前解決
            ResolveName();

            // 型推論
            foreach (Class cls in Layers) {

                TypeInference(cls);
            }

            DeepLearning();
        }
    }
}
