# Swarm Models System Documentation - Part 7

## Multi-Part Question Handling

The Multi-Part Question Handling system extends the swarm architecture by providing specialized support for processing and responding to queries that contain multiple distinct parts or questions. This system enables the swarm to detect when a query contains multiple parts, track which parts are addressed by different agents, and synthesize a complete answer that addresses all parts in a structured format.

### Core Components

1. **Query Structure Analysis**: Component for detecting and analyzing the structure of multi-part questions.
   - Identifies explicit parts like (a), (b), (c)
   - Detects numbered questions (1. 2. 3.)
   - Recognizes multiple questions with question marks
   - Analyzes the relationships between parts

2. **Part Contribution Tracking**: Component for tracking which parts of a multi-part question each agent addresses.
   - Monitors agent contributions to each part
   - Ensures all parts are covered
   - Identifies gaps in coverage
   - Guides agent focus to uncovered parts

3. **Enhanced Synthesis**: Component for synthesizing a complete answer that addresses all parts of a multi-part question.
   - Creates specialized prompts for multi-part questions
   - Ensures all parts are addressed in the final answer
   - Maintains the structure of the original question
   - Provides clear part labels in the response

### How It Works

1. When a query is received, the Query Structure Analysis component analyzes it to detect if it contains multiple parts.
2. If the query is identified as a multi-part question, the system extracts the individual parts and their structure.
3. As agents process the query, the Part Contribution Tracking component monitors which parts each agent addresses.
4. The Enhanced Synthesis component uses specialized prompts to ensure the final answer addresses all parts of the question.
5. The final answer is formatted to maintain the structure of the original question, with clear part labels for each response.

This approach ensures that all parts of a multi-part question are addressed comprehensively and in a structured format that makes it easy for users to find the answers to each part.

### Implementation Details

The Multi-Part Question Handling system has several key implementation features:

1. **Query Structure Analysis**: Detects and analyzes the structure of multi-part questions.
   ```python
   def analyze_query_structure(self, query):
       """Analyze the structure of a query to detect multi-part questions."""
       # Check for explicit parts like (a), (b), (c)
       explicit_parts = re.findall(r'\(([a-z])\)', query)
       if len(explicit_parts) > 1:
           return {
               "is_multi_part": True,
               "part_type": "explicit",
               "parts": explicit_parts,
               "part_prefix": "Part ({}):"
           }
       
       # Check for numbered questions like 1. 2. 3.
       numbered_parts = re.findall(r'(\d+)\.\s', query)
       if len(numbered_parts) > 1:
           return {
               "is_multi_part": True,
               "part_type": "numbered",
               "parts": numbered_parts,
               "part_prefix": "{}."
           }
       
       # Check for multiple question marks
       question_marks = query.count('?')
       if question_marks > 1:
           # Split by question marks
           questions = [q.strip() + '?' for q in query.split('?') if q.strip()]
           if len(questions) > 1:
               return {
                   "is_multi_part": True,
                   "part_type": "questions",
                   "parts": [str(i+1) for i in range(len(questions))],
                   "part_prefix": "Question {}:",
                   "questions": questions
               }
       
       # Not a multi-part question
       return {
           "is_multi_part": False
       }
   ```

2. **Part Contribution Tracking**: Tracks which parts of a multi-part question each agent addresses.
   ```python
   def track_part_contributions(self, agent_response, query_structure):
       """Track which parts of a multi-part question an agent addresses."""
       if not query_structure["is_multi_part"]:
           return {}
       
       part_contributions = {}
       
       for part in query_structure["parts"]:
           # Check if the agent addresses this part
           if query_structure["part_type"] == "explicit":
               # Look for explicit part references like "Part (a):" or "(a)"
               pattern = r'(?:Part\s*\({}(?:\)|:))|\(?{}(?:\)|:)'.format(part, part)
               if re.search(pattern, agent_response, re.IGNORECASE):
                   part_contributions[part] = True
           
           elif query_structure["part_type"] == "numbered":
               # Look for numbered part references like "1." or "Question 1:"
               pattern = r'(?:{}\.)|(?:Question\s*{}:)'.format(part, part)
               if re.search(pattern, agent_response, re.IGNORECASE):
                   part_contributions[part] = True
           
           elif query_structure["part_type"] == "questions":
               # For question mark-based parts, check if the agent addresses the content
               question_idx = int(part) - 1
               if question_idx < len(query_structure.get("questions", [])):
                   question = query_structure["questions"][question_idx]
                   # Extract key terms from the question
                   key_terms = self._extract_key_terms(question)
                   # Check if the agent response contains these key terms
                   if self._contains_key_terms(agent_response, key_terms):
                       part_contributions[part] = True
       
       return part_contributions
   
   def _extract_key_terms(self, question):
       """Extract key terms from a question."""
       # Remove stop words and extract nouns and verbs
       # This is a simplified implementation
       words = question.lower().split()
       stop_words = {"a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "in", "on", "at", "to", "for", "with", "by", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "from", "up", "down", "of", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"}
       key_terms = [word for word in words if word not in stop_words and len(word) > 3]
       return key_terms
   
   def _contains_key_terms(self, text, key_terms, threshold=0.5):
       """Check if text contains a sufficient number of key terms."""
       text = text.lower()
       matches = sum(1 for term in key_terms if term in text)
       return matches / len(key_terms) >= threshold if key_terms else False
   ```

3. **Enhanced Synthesis**: Synthesizes a complete answer that addresses all parts of a multi-part question.
   ```python
   def synthesize_multi_part_answer(self, agent_responses, query_structure):
       """Synthesize a complete answer for a multi-part question."""
       if not query_structure["is_multi_part"]:
           return self.synthesize_answer(agent_responses)
       
       # Create a specialized prompt for multi-part synthesis
       prompt = self._create_multi_part_synthesis_prompt(agent_responses, query_structure)
       
       # Generate the synthesized answer
       synthesized_answer = self._generate_with_prompt(prompt)
       
       # Ensure the answer has the correct structure
       structured_answer = self._structure_multi_part_answer(synthesized_answer, query_structure)
       
       return structured_answer
   
   def _create_multi_part_synthesis_prompt(self, agent_responses, query_structure):
       """Create a specialized prompt for multi-part synthesis."""
       prompt = "You are synthesizing answers to a multi-part question. The question has the following parts:\n\n"
       
       if query_structure["part_type"] == "explicit":
           for part in query_structure["parts"]:
               prompt += f"Part ({part})\n"
       elif query_structure["part_type"] == "numbered":
           for part in query_structure["parts"]:
               prompt += f"{part}.\n"
       elif query_structure["part_type"] == "questions":
           for i, question in enumerate(query_structure.get("questions", [])):
               prompt += f"Question {i+1}: {question}\n"
       
       prompt += "\nHere are the responses from different agents:\n\n"
       
       for i, response in enumerate(agent_responses):
           prompt += f"Agent {i+1}:\n{response}\n\n"
       
       prompt += "Synthesize a complete answer that addresses all parts of the question. Format your answer with clear part labels that match the original question structure. Ensure that each part is addressed comprehensively and accurately based on the agent responses."
       
       return prompt
   
   def _structure_multi_part_answer(self, answer, query_structure):
       """Ensure the answer has the correct structure for a multi-part question."""
       # Check if the answer already has the correct structure
       has_correct_structure = True
       
       for part in query_structure["parts"]:
           part_prefix = query_structure["part_prefix"].format(part)
           if part_prefix not in answer:
               has_correct_structure = False
               break
       
       if has_correct_structure:
           return answer
       
       # If the answer doesn't have the correct structure, restructure it
       structured_answer = "FINAL ANSWER:\n\n"
       
       if query_structure["part_type"] == "explicit" or query_structure["part_type"] == "numbered":
           # Split the answer into paragraphs
           paragraphs = [p.strip() for p in answer.split('\n\n') if p.strip()]
           
           # Assign paragraphs to parts
           for i, part in enumerate(query_structure["parts"]):
               part_prefix = query_structure["part_prefix"].format(part)
               if i < len(paragraphs):
                   structured_answer += f"{part_prefix} {paragraphs[i]}\n\n"
               else:
                   structured_answer += f"{part_prefix} No specific information provided for this part.\n\n"
       
       elif query_structure["part_type"] == "questions":
           # For question-based parts, try to match answers to questions
           questions = query_structure.get("questions", [])
           
           # Split the answer into paragraphs
           paragraphs = [p.strip() for p in answer.split('\n\n') if p.strip()]
           
           for i, question in enumerate(questions):
               part = str(i+1)
               part_prefix = query_structure["part_prefix"].format(part)
               
               # Extract key terms from the question
               key_terms = self._extract_key_terms(question)
               
               # Find the paragraph that best matches the question
               best_match = None
               best_score = 0
               
               for paragraph in paragraphs:
                   score = sum(1 for term in key_terms if term.lower() in paragraph.lower())
                   if score > best_score:
                       best_score = score
                       best_match = paragraph
               
               if best_match:
                   structured_answer += f"{part_prefix} {best_match}\n\n"
                   # Remove the matched paragraph to avoid reusing it
                   paragraphs.remove(best_match)
               else:
                   structured_answer += f"{part_prefix} No specific information provided for this question.\n\n"
       
       return structured_answer
   ```

### Integration with Swarm System

The Multi-Part Question Handling system integrates with the rest of the swarm system through:

1. **EnhancedFrontMan**: The system integrates with the EnhancedFrontMan to provide specialized handling for multi-part questions.
   - Sets the current query in the FrontMan generator
   - Detects multi-part questions and uses specialized prompts
   - Tracks part contributions for multi-part questions
   - Uses enhanced synthesis for multi-part questions

2. **RecurrentRegionManager**: The system integrates with the RecurrentRegionManager to coordinate agent activities for multi-part questions.
   - Manages agent interactions for multi-part questions
   - Tracks agent contributions to different parts
   - Ensures all parts are addressed
   - Coordinates the synthesis of the final answer

3. **SynthesisEnhancements**: The system integrates with the SynthesisEnhancements class for enhanced synthesis of multi-part answers.
   - Tracks part contributions from different agents
   - Enhances synthesis for multi-part questions
   - Analyzes query structure for multi-part detection
   - Ensures all parts are addressed in the final answer

### Example Usage

Here's an example of how the Multi-Part Question Handling system processes a multi-part question:

Input:
```
Consider Mars terraforming: (a) How long would it take to make Mars habitable? 
(b) What are the main challenges? (c) What technologies would be required?
```

Processing:
1. The system detects that this is a multi-part question with explicit parts (a), (b), and (c).
2. It extracts the individual parts and their structure.
3. As agents process the query, the system tracks which parts each agent addresses.
4. The Enhanced Synthesis component uses specialized prompts to ensure the final answer addresses all parts.
5. The final answer is formatted to maintain the structure of the original question, with clear part labels.

Output:
```
FINAL ANSWER:

Part (a): Making Mars habitable would likely take centuries to millennia, with estimates ranging from 100 years for basic habitability to thousands of years for Earth-like conditions.

Part (b): The main challenges include Mars' thin atmosphere, lack of magnetic field, extreme cold, radiation exposure, low gravity, lack of liquid water, and the enormous resources required.

Part (c): Required technologies would include atmospheric processors to release CO2 from the soil, methods to produce greenhouse gases, orbital mirrors to increase temperature, genetic engineering for extremophile organisms, nuclear or solar power generation, and advanced life support systems.
```

### Benefits

The Multi-Part Question Handling system provides several benefits to the swarm architecture:

1. **Improved User Experience**: Users can ask complex, multi-part questions and receive structured answers that address each part.
   - Clearer organization of responses
   - Easier to find answers to specific parts
   - More comprehensive coverage of complex questions
   - Better alignment with user expectations

2. **Enhanced Reasoning**: The system can break down complex questions into manageable parts, improving reasoning quality.
   - More focused reasoning on each part
   - Better handling of complex questions
   - Improved accuracy for multi-faceted queries
   - More systematic approach to complex problems

3. **Better Answer Completeness**: By tracking which parts have been addressed, the system ensures no parts of the question are missed.
   - Comprehensive coverage of all parts
   - Identification of gaps in coverage
   - Guidance for agents to address uncovered parts
   - More complete and thorough answers

4. **Structured Responses**: Answers maintain the structure of the original question, making them easier to understand.
   - Clear part labels in responses
   - Consistent formatting
   - Logical organization of information
   - Improved readability and comprehension

## Enhanced Query Classification

The Enhanced Query Classification system extends the swarm architecture by providing more sophisticated and accurate classification of user queries. This system enables the swarm to better understand the nature and requirements of each query, leading to more appropriate agent selection, improved reasoning strategies, and more accurate answers.

### Core Components

1. **QueryClassifier**: The main component responsible for classifying queries into different categories.
   - Identifies query types and domains
   - Detects specific problem patterns
   - Assesses query complexity
   - Provides confidence scores for classifications

2. **DomainDetector**: Component for detecting the domain of a query.
   - Identifies mathematical, scientific, historical, and other domains
   - Provides domain-specific confidence scores
   - Detects multi-domain queries
   - Guides domain-specific agent selection

3. **ComplexityAnalyzer**: Component for analyzing the complexity of a query.
   - Assesses reasoning complexity
   - Evaluates computational requirements
   - Identifies multi-step problems
   - Guides resource allocation

4. **PatternMatcher**: Component for matching queries to specific problem patterns.
   - Identifies common problem types
   - Matches queries to specialized solvers
   - Detects edge cases and special requirements
   - Guides specialized agent selection

### How It Works

1. When a query is received, the QueryClassifier analyzes it to determine its type, domain, complexity, and any specific patterns it matches.
2. The DomainDetector identifies the primary domain(s) of the query, such as mathematics, science, history, etc.
3. The ComplexityAnalyzer assesses the reasoning and computational complexity of the query.
4. The PatternMatcher checks if the query matches any specific problem patterns that might require specialized handling.
5. Based on the classification results, the system selects appropriate agents, reasoning strategies, and resources to process the query.
6. The classification information is also used to guide the synthesis of the final answer, ensuring it meets the specific requirements of the query type.

This approach ensures that each query is processed in the most appropriate way, leading to more accurate and relevant answers.

### Implementation Details

The Enhanced Query Classification system has several key implementation features:

1. **Query Type Classification**: Classifies queries into different types.
   ```python
   def classify_query_type(self, query):
       """Classify the query into different types."""
       # Check for calculation queries
       calculation_patterns = [
           r'\d+\s*[\+\-\*\/\^]\s*\d+',  # Basic arithmetic
           r'calculate',
           r'compute',
           r'evaluate',
           r'solve for',
           r'find the value',
           r'what is the result'
       ]
       
       for pattern in calculation_patterns:
           if re.search(pattern, query, re.IGNORECASE):
               return {
                   "type": "calculation",
                   "confidence": 0.9,
                   "subtype": self._detect_calculation_subtype(query)
               }
       
       # Check for factual queries
       factual_patterns = [
           r'what is',
           r'who is',
           r'where is',
           r'when did',
           r'how many',
           r'which',
           r'list',
           r'tell me about'
       ]
       
       for pattern in factual_patterns:
           if re.search(pattern, query, re.IGNORECASE):
               return {
                   "type": "factual",
                   "confidence": 0.8,
                   "subtype": self._detect_factual_subtype(query)
               }
       
       # Check for reasoning queries
       reasoning_patterns = [
           r'why',
           r'how does',
           r'explain',
           r'describe',
           r'compare',
           r'contrast',
           r'analyze',
           r'evaluate',
           r'what if',
           r'would',
           r'could'
       ]
       
       for pattern in reasoning_patterns:
           if re.search(pattern, query, re.IGNORECASE):
               return {
                   "type": "reasoning",
                   "confidence": 0.7,
                   "subtype": self._detect_reasoning_subtype(query)
               }
       
       # Check for creative queries
       creative_patterns = [
           r'create',
           r'generate',
           r'write',
           r'design',
           r'develop',
           r'imagine',
           r'suggest',
           r'come up with'
       ]
       
       for pattern in creative_patterns:
           if re.search(pattern, query, re.IGNORECASE):
               return {
                   "type": "creative",
                   "confidence": 0.7,
                   "subtype": self._detect_creative_subtype(query)
               }
       
       # Default to general query
       return {
           "type": "general",
           "confidence": 0.5,
           "subtype": "general"
       }
   ```

2. **Domain Detection**: Detects the domain of a query.
   ```python
   def detect_domain(self, query):
       """Detect the domain of a query."""
       domains = {}
       
       # Check for mathematical domain
       math_patterns = [
           r'\d+\s*[\+\-\*\/\^]\s*\d+',  # Basic arithmetic
           r'equation',
           r'formula',
           r'calculate',
           r'math',
           r'algebra',
           r'geometry',
           r'trigonometry',
           r'calculus',
           r'probability',
           r'statistics'
       ]
       
       math_score = 0
       for pattern in math_patterns:
           if re.search(pattern, query, re.IGNORECASE):
               math_score += 0.2
       
       if math_score > 0:
           domains['mathematical'] = min(1.0, math_score)
       
       # Check for scientific domain
       science_patterns = [
           r'science',
           r'physics',
           r'chemistry',
           r'biology',
           r'astronomy',
           r'geology',
           r'experiment',
           r'theory',
           r'hypothesis',
           r'molecule',
           r'atom',
           r'cell',
           r'organism',
           r'planet',
           r'star',
           r'galaxy'
       ]
       
       science_score = 0
       for pattern in science_patterns:
           if re.search(pattern, query, re.IGNORECASE):
               science_score += 0.2
       
       if science_score > 0:
           domains['scientific'] = min(1.0, science_score)
       
       # Check for historical domain
       history_patterns = [
           r'history',
           r'historical',
           r'ancient',
           r'medieval',
           r'century',
           r'war',
           r'revolution',
           r'civilization',
           r'empire',
           r'king',
           r'queen',
           r'president',
           r'leader',
           r'dynasty'
       ]
       
       history_score = 0
       for pattern in history_patterns:
           if re.search(pattern, query, re.IGNORECASE):
               history_score += 0.2
       
       if history_score > 0:
           domains['historical'] = min(1.0, history_score)
       
       # Add more domain checks as needed
       
       # If no specific domain is detected, default to general
       if not domains:
           domains['general'] = 0.5
       
       return domains
   ```

3. **Complexity Analysis**: Analyzes the complexity of a query.
   ```python
   def analyze_complexity(self, query):
       """Analyze the complexity of a query."""
       complexity = {
           "reasoning": 0.0,  # 0.0 to 1.0, higher means more complex reasoning
           "computational": 0.0,  # 0.0 to 1.0, higher means more computational resources
           "multi_step": False,  # Whether the query requires multiple steps
           "specialized": False  # Whether the query requires specialized knowledge
       }
       
       # Check for indicators of reasoning complexity
       reasoning_indicators = [
           r'why',
           r'how does',
           r'explain',
           r'analyze',
           r'evaluate',
           r'compare',
           r'contrast',
           r'relationship',
           r'connection',
           r'implication',
           r'consequence',
           r'effect',
           r'cause',
           r'reason'
       ]
       
       reasoning_score = 0
       for indicator in reasoning_indicators:
           if re.search(indicator, query, re.IGNORECASE):
               reasoning_score += 0.1
       
       complexity["reasoning"] = min(1.0, reasoning_score)
       
       # Check for indicators of computational complexity
       computational_indicators = [
           r'\d+\s*[\+\-\*\/\^]\s*\d+',  # Basic arithmetic
           r'calculate',
           r'compute',
           r'solve',
           r'equation',
           r'formula',
           r'algorithm',
           r'optimization',
           r'simulation',
           r'model'
       ]
       
       computational_score = 0
       for indicator in computational_indicators:
           if re.search(indicator, query, re.IGNORECASE):
               computational_score += 0.1
       
       complexity["computational"] = min(1.0, computational_score)
       
       # Check for indicators of multi-step problems
       multi_step_indicators = [
           r'steps',
           r'process',
           r'procedure',
           r'method',
           r'approach',
           r'first.*then',
           r'after that',
           r'finally',
           r'sequence',
           r'order'
       ]
       
       for indicator in multi_step_indicators:
           if re.search(indicator, query, re.IGNORECASE):
               complexity["multi_step"] = True
               break
       
       # Check for indicators of specialized knowledge
       specialized_indicators = [
           r'technical',
           r'specialized',
           r'expert',
           r'advanced',
           r'professional',
           r'specific',
           r'detailed',
           r'in-depth',
           r'comprehensive'
       ]
       
       for indicator in specialized_indicators:
           if re.search(indicator, query, re.IGNORECASE):
               complexity["specialized"] = True
               break
       
       return complexity
   ```

4. **Pattern Matching**: Matches queries to specific problem patterns.
   ```python
   def match_patterns(self, query):
       """Match the query to specific problem patterns."""
       patterns = []
       
       # Check for logic puzzle patterns
       logic_puzzle_patterns = [
           r'logic puzzle',
           r'logic problem',
           r'knights and knaves',
           r'truth teller',
           r'liar',
           r'if.*then',
           r'deduction',
           r'inference',
           r'syllogism'
       ]
       
       for pattern in logic_puzzle_patterns:
           if re.search(pattern, query, re.IGNORECASE):
               patterns.append({
                   "type": "logic_puzzle",
                   "confidence": 0.8,
                   "subtype": self._detect_logic_puzzle_subtype(query)
               })
               break
       
       # Check for state planning patterns
       state_planning_patterns = [
           r'river crossing',
           r'wolf.*goat.*cabbage',
           r'missionaries.*cannibals',
           r'tower of hanoi',
           r'water jug',
           r'state',
           r'planning',
           r'sequence',
           r'steps',
           r'move',
           r'transport'
       ]
       
       for pattern in state_planning_patterns:
           if re.search(pattern, query, re.IGNORECASE):
               patterns.append({
                   "type": "state_planning",
                   "confidence": 0.8,
                   "subtype": self._detect_state_planning_subtype(query)
               })
               break
       
       # Check for mathematical problem patterns
       math_problem_patterns = [
           r'equation',
           r'solve for',
           r'find the value',
           r'calculate',
           r'compute',
           r'arithmetic',
           r'algebra',
           r'geometry',
           r'calculus',
           r'probability',
           r'statistics'
       ]
       
       for pattern in math_problem_patterns:
           if re.search(pattern, query, re.IGNORECASE):
               patterns.append({
                   "type": "mathematical",
                   "confidence": 0.8,
                   "subtype": self._detect_math_problem_subtype(query)
               })
               break
       
       # Add more pattern checks as needed
       
       return patterns
   ```

### Integration with Swarm System

The Enhanced Query Classification system integrates with the rest of the swarm system through:

1. **RecurrentRegionManager**: The system integrates with the RecurrentRegionManager to guide agent selection and coordination.
   - Provides classification information to the manager
   - Guides agent selection based on query type and domain
   - Informs resource allocation based on complexity
   - Supports specialized handling for specific patterns

2. **Progressive Agent Loader**: The system integrates with the Progressive Agent Loader to optimize agent selection.
   - Guides the selection of specialized agents
   - Informs the progressive loading process
   - Helps determine which agents to add first
   - Supports early stopping based on query requirements

3. **EnhancedFrontMan**: The system integrates with the EnhancedFrontMan to guide the synthesis process.
   - Provides classification information to the FrontMan
   - Guides the synthesis of the final answer
   - Informs the selection of synthesis strategies
   - Supports specialized formatting for different query types

### Example Usage

Here's an example of how the Enhanced Query Classification system processes a query:

Input:
```
Solve the quadratic equation x^2 - 5x + 6 = 0 and explain the steps.
```

Processing:
1. The system classifies this as a calculation query with a mathematical domain.
2. It detects that this is a quadratic equation problem.
3. It assesses the complexity as moderate, requiring computational resources and multiple steps.
4. Based on this classification, the system selects appropriate agents, including mathematical specialists.
5. The classification information guides the synthesis of the final answer, ensuring it includes both the solution and the explanation of steps.

Output:
```
FINAL ANSWER:

To solve the quadratic equation x^2 - 5x + 6 = 0, I'll use the factoring method.

Step 1: Identify the coefficients.
a = 1, b = -5, c = 6

Step 2: Find two numbers that multiply to give 'c' (6) and add up to 'b' (-5).
The numbers -2 and -3 multiply to give 6 and add up to -5.

Step 3: Rewrite the middle term using these numbers.
x^2 - 5x + 6 = x^2 - 2x - 3x + 6 = x(x - 2) - 3(x - 2) = (x - 2)(x - 3)

Step 4: Set each factor equal to zero and solve.
x - 2 = 0 → x = 2
x - 3 = 0 → x = 3

Therefore, the solutions to the equation x^2 - 5x + 6 = 0 are x = 2 and x = 3.
```

### Benefits

The Enhanced Query Classification system provides several benefits to the swarm architecture:

1. **Improved Agent Selection**: The system enables more appropriate agent selection based on query characteristics.
   - More relevant agent expertise
   - Better alignment with query requirements
   - More efficient use of specialized agents
   - Improved overall performance

2. **Optimized Resource Allocation**: The system helps allocate computational resources more efficiently based on query complexity.
   - More resources for complex queries
   - Fewer resources for simple queries
   - Better balance of resource usage
   - Improved system efficiency

3. **Enhanced Reasoning Strategies**: The system guides the selection of appropriate reasoning strategies for different query types.
   - More appropriate reasoning approaches
   - Better handling of specialized problems
   - Improved accuracy for different query types
   - More efficient problem-solving

4. **Better Answer Quality**: The system helps ensure that answers meet the specific requirements of different query types.
   - More relevant and accurate answers
   - Better formatting for different query types
   - More comprehensive coverage of query requirements
   - Improved user satisfaction