
import unittest
import requests
import json
import spacy



class TextualAnalysis:
    # This method takes in an article and author name
    def query(self, prompt, author):
        API_KEY = ""
        request_data = {
            "model": "gpt-3.5-turbo", 
            "messages": [{"role": "system", "content": "You will take on the persona of the author: " + author + ", you will translate any text coming from the user into the" + author + "'s specific style. Basically, you will rewrite the text provided in that new style."}, {"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 700,
        }
        
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", 
                                headers=headers, 
                                json=request_data)
                                
        return response.json()




class TestTextualAnalysis(unittest.TestCase):
    analyzer = TextualAnalysis()
    # Instantiate the TextualAnalysis class

    # article
    prompt = """
        It's interesting how human beings have experienced and observed a phenomena such as Antifragility for such a long time but lacked the word to describe it. My last name Magtibay is derived from the word Tibay or Matibay, a tagalog word that translates directly into the English words durable or robust. Magtibay then becomes enduring/robust people, but then again we have no Tagalog word for describing antifragility. I suppose with all our progress with natural language we still lack the words to articulate some of our experiences and observations. Venkatesh Rao talks about math and formalism being a crude projection (but coldly precise) of our fuzzy notion of "what I want to get at". Making language, be it natural language or mathematics, as a bottleneck of communication. Makes me think how do we, as a culture, process in creating a word that describes a bit of universality that most people would understand?

        Reminds me of the conversation I had with my professor on the inherent difficulty of high-level physics. He suggested that maybe as a species we just lacked the mental horsepower to comprehend the weirdness of such subjects like quantum mechanics. Like penguins trying to understand how ballistics work (but maybe they do, don't want to doubt penguins here). I suggested maybe language is limiting us and he replied "Maybe, but we speak math very well." and then coming back to his argument of our probable inadequacy. But now I think, maybe we are just still lacking the words (mathematical) to describe the weirdness of things and as we progress knowledge as a species we would stumble upon the right words. But maybe that's the idealist in me.

        """
    throwaway_text = """
        Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem. Ut enim ad minima veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur? Quis autem vel eum iure reprehenderit qui in ea voluptate velit esse quam nihil molestiae consequatur, vel illum qui dolorem eum fugiat quo voluptas nulla pariatur?
        """

    author = "Frank Herbert"         
    simulated_text = analyzer.query(prompt, author)


    def test_generate_simulated_text(self):

        # Output the result or "fail"
        print(self.simulated_text['choices'][0]['message']['content']) if self.simulated_text else print("test failed")

        # Ensure that the simulated text is not None (i.e., the algorithm did not fail)
        self.assertIsNotNone(self.simulated_text)

    def evaluate_test(self):
        from sklearn.feature_extraction.text import CountVectorizer
        import pandas as pd
        from sklearn.metrics.pairwise import cosine_similarity
        import spacy


        ############### cosine test
        augmented_text = str(self.simulated_text)
        documents = [self.prompt, augmented_text, self.throwaway_text]

        # count vectorizer turns text into a maatrix of token counts
        count_vectorizer = CountVectorizer()

        sparse_matrix = count_vectorizer.fit_transform(documents)
        
        
        doc_term_matrix = sparse_matrix.todense()
        df = pd.DataFrame(
        doc_term_matrix,
        columns=count_vectorizer.get_feature_names_out(),
        index=["original_text", "augmented_text", "throwaway_text"],
        )
        # print(df)
        out_matrix = cosine_similarity(df, df)
        print(out_matrix)
        if out_matrix[0][1] < 0.5:
            print("Cosine test fail.")
        else:
            print("Passed cosine test.")

        ################# semantic similarity
        nlp = spacy.load('en_core_web_lg')

        author_doc = nlp(self.prompt)
        generated_doc = nlp(augmented_text)
        semantic_similarity = author_doc.similarity(generated_doc)
        print("Semantic Similarity:", semantic_similarity)
        if semantic_similarity > 0.6:
            print("Passed semantic test.")
        else:
            print("Semantic test fail.")

        ################ length similarity
        author_length = len((self.prompt).split())
        generated_length = len(augmented_text.split())
        length_similarity = min(author_length, generated_length) / max(author_length, generated_length)
        print("Length Similarity:", length_similarity)
        if length_similarity > 0.6:
            print("Passed length test.")
        else:
            print("Length test fail.")

        




# Run the test
unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestTextualAnalysis))


TestTextualAnalysis().evaluate_test()